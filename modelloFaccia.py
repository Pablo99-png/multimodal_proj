import os, random, time, math
from itertools import repeat
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from PIL import Image

# reproducibility -------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------
# 2.  Building blocks
# -----------------------------------------------------------
def _ntuple(n):
    def parse(x):
        return tuple(repeat(x, n)) if not isinstance(x, (list, tuple)) else x
    return parse
to_2tuple = _ntuple(2)


class PatchMerging(nn.Module):
    """
    Swin-style down-sampling.

    Input : (B, N, C)  where N = H × W
    Output: (B, N/4, C_out) with H & W halved.
    """

    def __init__(self, dim_in: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim_out or dim_in * 2
        self.norm = nn.LayerNorm(4 * dim_in)
        self.reduction = nn.Linear(4 * dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, "Token length doesn't match H*W"

        x = x.view(B, H, W, C)
        # gather 2 × 2 neighbours
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)     # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                    # (B, N/4, 4C)
        x = self.norm(x)
        x = self.reduction(x)                       # (B, N/4, C_out)
        return x


class TinyViTBlock(nn.Module):
    """
    **Light** substitute for Swin/TinyViT window attention.
    Keeps tensor shapes identical; swap with real implementation later.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.ModuleList([layer for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ArcMarginProduct(nn.Module):
    """
    ArcFace: additive angular margin.\n
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.s = s
        self.m = m

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x)
        W = F.normalize(self.weight)
        logits = F.linear(x, W)                     # cosθ
        θ = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        logits = self.s * torch.cos(θ + self.m)     # cos(θ + m)
        return logits


# -----------------------------------------------------------
# 3.  FaceTinyViT model (96 → 192 channels with CLS projection)
# -----------------------------------------------------------
class FaceTinyViT(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 512):
        super().__init__()

        # ---- Patch stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.GELU()
        )

        # stride-4 patch embedding → 96-d tokens
        self.patch_embed = nn.Conv2d(32, 96, kernel_size=4, stride=4, bias=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, 96))

        # ---- Stage 1 ----
        self.stage1   = TinyViTBlock(dim=96, depth=2, num_heads=3)
        self.down1    = PatchMerging(96, 192)           # 96 → 192
        self.cls_proj1 = nn.Linear(96, 192)             # project CLS

        # ---- Stage 2 ----
        self.stage2   = TinyViTBlock(dim=192, depth=2, num_heads=6)

        # ---- Heads ----
        self.norm         = nn.LayerNorm(192)
        self.head_embed   = nn.Linear(192, emb_dim, bias=False)
        self.arcface      = ArcMarginProduct(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_embedding: bool = False,
    ):
        # ---- Stem & patch embedding ----
        x = self.stem(x)                 # B, 32, 56, 56   (for 112×112 input)
        B, _, H, W = x.shape
        x = self.patch_embed(x)          # B, 96, 14, 14
        H //= 4; W //= 4

        x = x.flatten(2).transpose(1, 2)  # B, N, 96
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat((cls, x), 1)      # B, 1+N, 96

        # ---- Stage 1 ----
        x = self.stage1(x)
        cls_tok, feat = x[:, :1], x[:, 1:]
        feat = self.down1(feat, H, W)     # B, N/4, 192
        H //= 2; W //= 2
        cls_tok = self.cls_proj1(cls_tok) # B, 1, 192
        x = torch.cat((cls_tok, feat), 1) # B, 1+N/4, 192

        # ---- Stage 2 ----
        x = self.stage2(x)

        # ---- Head ----
        cls_out   = self.norm(x[:, 0])
        embedding = F.normalize(self.head_embed(cls_out), p=2, dim=1)

        if return_embedding or labels is None:
            return embedding

        logits = self.arcface(embedding, labels)
        return logits, embedding


# -----------------------------------------------------------
# 4.  Dataset & loaders
# -----------------------------------------------------------
DATA_ROOT = Path('/Users/paolocursi/Desktop/biometrics/archive/train')
IMG_SIZE  = 112
MEAN_STD  = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), value='random'),
    ])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(*MEAN_STD),
])

def load():
    

    
    full_ds      = datasets.ImageFolder(DATA_ROOT, transform=train_tfms)
    num_classes  = len(full_ds.classes)
    val_fraction = 0.1
    val_len      = int(len(full_ds) * val_fraction)
    train_len    = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_ds.dataset.transform = val_tfms

    BATCH_SIZE = 128
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )


    # -----------------------------------------------------------
    # 5.  Training utilities
    # -----------------------------------------------------------
    device   = 'mps'
    model    = FaceTinyViT(num_classes=num_classes, emb_dim=512).to(device)

    # Load pre-trained weights if available
    weights_path = '/Users/paolocursi/Desktop/multimodal/tinyvit_face_best.pth'

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
        print("Weights loaded successfully")
    else:
        print("No pre-trained weights found, starting from scratch")
    return model



def compute_ids(k,model):
    # Define the range of folders to process
    start_folder = 100  # Start from n000002
    end_folder = 480  # Process 100 folders

    # Storage for the processed tensors
    identity_tensors = {}

    for folder_id in range(start_folder, end_folder + 1):
        folder_name = f"n{folder_id:06d}"  # Format as n000002, n000003, etc.
        folder_path = DATA_ROOT / folder_name
        
        if not folder_path.exists():
            continue
        
        # Get all image files in the folder
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        
        if len(image_files) < k:
            print(f"Folder {folder_name} has insufficient images ({len(image_files)}), skipping...")
            continue
        
        # Load and transform images to tensors
        tensors = []
        for img_path in image_files[:k]:  # Take only first k images
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = val_tfms(img)  # Apply validation transforms
                tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(tensors) < k:
            print(f"Folder {folder_name} has insufficient valid images, skipping...")
            continue
        
        # Stack tensors
        image_batch = torch.stack(tensors).to('mps')  # Shape: (k, 3, 112, 112)
        
        # Process in batches
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(image_batch), batch_size):
            batch = image_batch[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = model(batch, return_embedding=True)
            embeddings.append(batch_embeddings)
            del batch, batch_embeddings
            torch.cuda.empty_cache()
        
        embeddings = torch.cat(embeddings, dim=0)
        mean_embedding = embeddings.mean(dim=0)
        del image_batch, embeddings
        torch.cuda.empty_cache()
        
        # Store the result
        identity_tensors[folder_name] = mean_embedding
    
    return identity_tensors

def compute_emb(model, face):
    #if type(face) is not np.ndarray:
    face_tensor = val_tfms(face).unsqueeze(0).to('mps')
    # Get embedding
    with torch.no_grad():
        embedding = model(face_tensor, return_embedding=True).squeeze(0)

    return embedding  # Convert to numpy array for easier handling