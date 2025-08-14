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

print(f"device: {device}")
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



no_users = 1
batch_size = 128
modalities = ["lidar", "RGB", "mmwave"]

dataset_dir = "dataset/"
train_loaders = []
test_loaders = []
val_loaders = []

for user_id in range(no_users):
    train_dir = dataset_dir  + f'train/user_{user_id}.csv'
    val_dir = dataset_dir  + f'test/user_{user_id}.csv'
    weight_dir = dataset_dir  + f'weights/user_{user_id}_weights.pt'
    # Load from file
    sample_weights_loaded = torch.load(weight_dir, weights_only=True)
    
    # Create sampler
    sampler = WeightedRandomSampler(sample_weights_loaded, num_samples=len(sample_weights_loaded), replacement=True)
    
    train_dataset = FutureClearWindowDataset(csv_path=train_dir, window_length=4, T_f=2)
    val_dataset = FutureClearWindowDataset(csv_path=val_dir, window_length=4, T_f=2)
    
    
    train_loaders.append(DataLoader(train_dataset,
                              batch_size=batch_size, sampler=sampler))
    val_loaders.append(DataLoader(val_dataset,
                            batch_size=batch_size,
                            #num_workers=8,
                            shuffle=False))



print("Dataloader finished")



# model1 = RGBPredictionCNN(input_channels=4, feature_dim=128)

# for batch in train_loaders[0]:
#     modalitiy_input = {mod: batch[mod].to(device) for mod in modalities}
#     print(modalitiy_input["lidar"].shape, modalitiy_input["RGB"].shape, modalitiy_input["mmwave"].shape)
#     y = model1(modalitiy_input["RGB"])
#     break

print("Finished one batch")


# 3 nodes (modalities)
num_modalities = 3
embed_dim = 128
encoders = nn.ModuleDict({"lidar": SymmetricLiDARCNN(input_channels=2, feature_dim=128),
             "RGB": RGBPredictionCNN(input_channels=4, feature_dim=128),
             "mmwave": mmWaveSCRNet(input_channels=4, feature_dim=128)}).to(device)


modality_node_dict = {0: "lidar", 1: "RGB", 2: "mmwave"}

# ====== 3. Restriction maps P_{i->e} and Dual maps Q_{e->i} ======
edges = [(0 , 1),
         (0, 2),
         (1, 2)]
edge_dim = 64  # shared comparison space

restrictions = nn.ParameterDict()  # P maps
duals = nn.ParameterDict()         # Q maps

for (i,j) in edges:
    # P: node -> edge
    restrictions[f"{i}->{i}-{j}"] = nn.Parameter(torch.randn(edge_dim, embed_dim) * 0.1).to(device)
    restrictions[f"{j}->{i}-{j}"] = nn.Parameter(torch.randn(edge_dim, embed_dim) * 0.1).to(device)

    # Q: edge -> node
    duals[f"{i}-{j}->{i}"] = nn.Parameter(torch.randn(embed_dim, edge_dim) * 0.1).to(device)
    duals[f"{i}-{j}->{j}"] = nn.Parameter(torch.randn(embed_dim, edge_dim) * 0.1).to(device)


# ====== 4. Loss functions ======
def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)

def contrastive_loss(p_i, p_j, tau=0.1):
    # p_i, p_j: [B, D]
    sim_ij = cosine_sim(p_i, p_j) / tau
    labels = torch.arange(p_i.size(0)).to(device)
    # Symmetric InfoNCE
    loss_i = F.cross_entropy(sim_ij, labels)
    loss_j = F.cross_entropy(sim_ij.t(), labels)
    return 0.5 * (loss_i + loss_j)

def laplacian_loss(p_i, p_j):
    return ((p_i - p_j)**2).sum(dim=1).mean()

def reconstruction_loss(h_i, p_i, Q_ei):
    # Reconstruct node embedding from edge embedding
    # p_i: node->edge embedding [B, edge_dim]
    # Q_ei: edge->node map [embed_dim, edge_dim]
    recon = p_i @ Q_ei.t()  # [B, embed_dim]
    return F.mse_loss(recon, h_i)



optimizer = torch.optim.Adam(
    list(encoders.parameters()) + list(restrictions.parameters()) + list(duals.parameters()),
    lr=1e-4
)

lambda_lap = 1.0
beta_contrast = 1.0
gamma_recon = 0.1  # weight for reconstruction loss

for epoch in range(50):  # small demo
    for i, batch in zip(range(20), train_loaders[0]):
        modalitiy_input = {mod: batch[mod].to(device) for mod in modalities}
        batch_size = modalitiy_input[modality_node_dict[0]].size(0)

        # Local embeddings h_i
        h = {mod_id: encoders[modality_node_dict[mod_id]](modalitiy_input[modality_node_dict[mod_id]]) for mod_id in modality_node_dict}  # list of [B, embed_dim]

        total_loss = 0.0

        # Loop over edges for sheaf contrastive + Laplacian + Reconstruction
        for (i,j) in edges:
            P_i = restrictions[f"{i}->{i}-{j}"]
            P_j = restrictions[f"{j}->{i}-{j}"]
            Q_ei = duals[f"{i}-{j}->{i}"]
            Q_ej = duals[f"{i}-{j}->{j}"]

            p_i = (h[i] @ P_i.t())   # [B, edge_dim]
            p_j = (h[j] @ P_j.t())   # [B, edge_dim]

            lap_loss = laplacian_loss(p_i, p_j)
            contrast_loss = contrastive_loss(p_i, p_j)

            # Reconstruction from edge back to nodes
            recon_loss_i = reconstruction_loss(h[i], p_i, Q_ei)
            recon_loss_j = reconstruction_loss(h[j], p_j, Q_ej)

            
            total_loss += (
                lambda_lap * lap_loss +
                beta_contrast * contrast_loss +
                gamma_recon * (recon_loss_i + recon_loss_j)
            )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")



model_dir = "models/"


def save_P_Q_all(edges, restrictions, duals, save_path=model_dir + "P_Q_maps.pt"):
    pq_data = {}
    
    for (i, j) in edges:
        # Restriction maps
        pq_data[f"P_{i}_to_{i}-{j}"] = restrictions[f"{i}->{i}-{j}"].detach().cpu()
        pq_data[f"P_{j}_to_{i}-{j}"] = restrictions[f"{j}->{i}-{j}"].detach().cpu()

        # Reconstruction maps
        pq_data[f"Q_{i}-{j}_to_{i}"] = duals[f"{i}-{j}->{i}"].detach().cpu()
        pq_data[f"Q_{i}-{j}_to_{j}"] = duals[f"{i}-{j}->{j}"].detach().cpu()

    torch.save(pq_data, save_path)
    print(f"Saved P and Q maps for {len(edges)} edges to {save_path}")



# Save all encoders' state_dict
encoder_states = {mod: encoders[mod].state_dict() for mod in encoders}
torch.save(encoder_states, model_dir + "trained_encoders.pt")
print("Saved all encoder models to trained_encoders.pt")


save_P_Q_all(edges, restrictions, duals)
