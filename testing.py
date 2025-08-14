import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import pandas as pd
import random
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from models import SymmetricLiDARCNN, RGBPredictionCNN, mmWaveSCRNet
from data_set import FutureClearWindowDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the encoder dict with the same architectures
loaded_encoders = nn.ModuleDict({
    "lidar": SymmetricLiDARCNN(input_channels=2, feature_dim=128),
    "RGB": RGBPredictionCNN(input_channels=4, feature_dim=128),
    "mmwave": mmWaveSCRNet(input_channels=4, feature_dim=128)
})





model_dir = "models/"
# Load state dicts
encoder_states = torch.load(model_dir + "trained_encoders.pt")
for mod in loaded_encoders:
    loaded_encoders[mod].load_state_dict(encoder_states[mod])
    loaded_encoders[mod].to(device)
    loaded_encoders[mod].eval()  # optional if only for inference


def load_encoders_P_Q(edges, save_path=model_dir + "P_Q_maps.pt", device=device):
    """
    Load saved restriction maps (P) and reconstruction maps (Q) into ParameterDicts.

    Args:
        edges: list of (i, j) tuples for graph edges
        save_path: path to the saved .pt file
        device: device to load the tensors on ("cpu" or "cuda")

    Returns:
        restrictions: nn.ParameterDict containing P maps
        duals: nn.ParameterDict containing Q maps
    """
    pq_data = torch.load(save_path, map_location=device)

    restrictions = nn.ParameterDict()
    duals = nn.ParameterDict()

    for (i, j) in edges:
        # Restore P maps
        restrictions[f"{i}->{i}-{j}"] = nn.Parameter(pq_data[f"P_{i}_to_{i}-{j}"].to(device))
        restrictions[f"{j}->{i}-{j}"] = nn.Parameter(pq_data[f"P_{j}_to_{i}-{j}"].to(device))

        # Restore Q maps
        duals[f"{i}-{j}->{i}"] = nn.Parameter(pq_data[f"Q_{i}-{j}_to_{i}"].to(device))
        duals[f"{i}-{j}->{j}"] = nn.Parameter(pq_data[f"Q_{i}-{j}_to_{j}"].to(device))

    print(f"Loaded {len(restrictions)} P maps and {len(duals)} Q maps from {save_path}")
    return restrictions, duals

modality_node_dict = {0: "lidar", 1: "RGB", 2: "mmwave"}

# ====== 3. Restriction maps P_{i->e} and Dual maps Q_{e->i} ======
edges = [(0 , 1),
         (0, 2),
         (1, 2)]
edge_dim = 64  # shared comparison space




restrictions, duals = load_encoders_P_Q(edges, save_path=model_dir + "P_Q_maps.pt", device=device)


print("all models loaded succefully")