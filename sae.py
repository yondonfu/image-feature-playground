import torch
import os
import importlib
from typing import Union
from huggingface_hub import snapshot_download


# SAE related code from https://colab.research.google.com/drive/1m6IRqKaRUWRL-d2PC82nslb7kmAgjoKV?usp=sharing
def load_sae(model_id: str, device: Union[str, torch.device], dtype: torch.dtype):
    # If the model is cached already this should just return the local file path
    sae_file_path = snapshot_download(model_id)
    model_path = os.path.join(sae_file_path, "model.py")
    weights_path = os.path.join(sae_file_path, "sparse_autoencoder_128.pth")

    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    SAE = getattr(model_module, "SparseAutoencoder")

    embedding_dim = 1280
    hidden_dim = embedding_dim * 128
    model = SAE(embedding_dim, hidden_dim).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model
