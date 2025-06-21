from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from resemblyzer import VoiceEncoder, preprocess_wav
import torch

# ---------------------------------------------------------------------
# 1)  Initialise encoder once (weights are auto-downloaded on first run)
# ---------------------------------------------------------------------


def load():
    device = torch.device('mps')
    return VoiceEncoder(device=device)

def compute_ids(k,encoder):
    # Create a dictionary to store embeddings for each identity
    base_path = Path('/Users/paolocursi/Desktop/multimodal/LibriSpeech/dev-clean')
    identity_embeddings_split = {}

    for identity_folder in base_path.iterdir():
        if identity_folder.is_dir():
            identity_id = identity_folder.name
            embeddings_list = []
            
            # Get all audio files from all subfolders of this identity
            for subfolder in identity_folder.iterdir():
                if subfolder.is_dir():
                    for audio_file in subfolder.glob('*.flac'):
                        try:
                            wav_data = preprocess_wav(str(audio_file))
                            embedding = encoder.embed_utterance(wav_data)
                            embeddings_list.append(embedding)
                        except Exception as e:
                            print(f"Error processing {audio_file}: {e}")
            
            # If we have embeddings, calculate averagesa
            if embeddings_list:
                # Get first 10 and second 10 embeddings
                first_10 = embeddings_list[:k]
                second_10 = embeddings_list[k:2*k] if len(embeddings_list) > 10 else []
                
                # Calculate averages
                avg_first_10 = np.mean(first_10, axis=0) if first_10 else None
                avg_second_10 = np.mean(second_10, axis=0) if second_10 else None
                
                identity_embeddings_split[identity_id] = (avg_first_10, avg_second_10)
            

    return identity_embeddings_split

def compute_emb(model,path):
    wav_data = preprocess_wav(str(path))
    return model.embed_utterance(wav_data)