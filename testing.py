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
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





# Recreate the encoder dict with the same architectures
encoders = nn.ModuleDict({
    "lidar": SymmetricLiDARCNN(input_channels=2, feature_dim=128),
    "RGB": RGBPredictionCNN(input_channels=4, feature_dim=128),
    "mmwave": mmWaveSCRNet(input_channels=4, feature_dim=128)
})



no_users = 1
batch_size = 128
modalities = ["lidar", "RGB", "mmwave"]
modality_node_dict = {0: "lidar", 1: "RGB", 2: "mmwave"}

# ====== 3. Restriction maps P_{i->e} and Dual maps Q_{e->i} ======
edges = [(0 , 1),
         (0, 2),
         (1, 2)]


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
    test_loaders.append(DataLoader(val_dataset,
                            batch_size=batch_size,
                            #num_workers=8,
                            shuffle=False))



print("Dataloader finished")





model_dir = "models/"
# Load state dicts
encoder_states = torch.load(model_dir + "trained_encoders.pt", map_location=device)
for mod in encoders:
    print(mod)
    encoders[mod].load_state_dict(encoder_states[mod])
    encoders[mod].to(device)
    encoders[mod].eval()  # optional if only for inference


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





# ====== 2. Encode + Optionally Reconstruct via Dual Maps ======
def encode_modality(modality_idx, test_loader, use_reconstruction=False):
    enc = encoders[modality_node_dict[modality_idx]].eval()
    embs, labels = [], []
    with torch.no_grad():
        for i, batch in zip(tqdm(range(20)), test_loader):
            modalitiy_input = {mod: batch[mod].to(device) for mod in modalities}
            y = batch["label"].to(device)
            h = enc(modalitiy_input[modality_node_dict[modality_idx]])  # [B, embed_dim]

            if use_reconstruction:
                # Aggregate reconstructions from all edges touching this node
                reconstructions = []
                for (i, j) in edges:
                    if modality_idx in (i, j):
                        edge_key = f"{i}-{j}"
                        node_key = f"{modality_idx}->{edge_key}"
                        dual_key = f"{edge_key}->{modality_idx}"
                        P = restrictions[node_key]    # [edge_dim, embed_dim]
                        Q = duals[dual_key]           # [embed_dim, edge_dim]

                        edge_emb = h @ P.t()          # project to edge
                        node_recon = edge_emb @ Q.t() # reconstruct node
                        reconstructions.append(node_recon)

                if len(reconstructions) > 0:
                    h = sum(reconstructions) / len(reconstructions)

            h = F.normalize(h, dim=-1)
            embs.append(h)
            labels.append(y)
    return torch.cat(embs), torch.cat(labels)


# ====== 3. Zero-Shot Learning Evaluation ======
def zero_shot_eval(test_loader, mod_idx=0, use_reconstruction=False):
    print(f"\n[Zero-Shot Eval] Using modality {mod_idx} embeddings "
          f"{'(reconstructed)' if use_reconstruction else '(raw)'}")
    # Encode all test data
    test_embs, test_labels = encode_modality(mod_idx, test_loader, use_reconstruction)

    # Compute class centroids
    num_classes = 2
    centroids = []
    for c in range(num_classes):
        class_embs = test_embs[test_labels == c]
        centroids.append(class_embs.mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), dim=-1)  # [10, D]

    # Classify by cosine similarity
    sims = test_embs @ centroids.t()  # [N, 10]
    preds = sims.argmax(dim=1)
    acc = (preds == test_labels.to(device)).float().mean().item()
    print(f"Zero-Shot Accuracy (modality {mod_idx}): {acc*100:.2f}%")
    return acc


# ====== 4. Cross-Modal Retrieval ======
def cross_modal_retrieval(test_loader, query_mod=0, gallery_mod=1, top_k=(1,5,10), use_reconstruction=False):
    print(f"\n[Cross-Modal Retrieval] Query: {query_mod} -> Gallery: {gallery_mod} "
          f"{'(reconstructed)' if use_reconstruction else '(raw)'}")

    q_embs, q_labels = encode_modality(query_mod, test_loader, use_reconstruction)
    g_embs, g_labels = encode_modality(gallery_mod, test_loader, use_reconstruction)

    # Compute cosine similarity
    sim = q_embs @ g_embs.t()
    ranks = sim.argsort(dim=1, descending=True)

    recalls = {k: 0 for k in top_k}
    for i, label in enumerate(q_labels):
        retrieved_labels = g_labels[ranks[i]]
        for k in top_k:
            if label in retrieved_labels[:k]:
                recalls[k] += 1

    for k in top_k:
        recalls[k] /= len(q_labels)
        print(f"Recall@{k}: {recalls[k]*100:.2f}%")
    return recalls


# ====== 5. Run evaluations ======
#if __name__ == "__main__":
#    # Zero-shot on raw and reconstructed embeddings
#    zero_shot_eval(test_loader=train_loaders[0], mod_idx=1, use_reconstruction=False)
#    zero_shot_eval(test_loader=train_loaders[0], mod_idx=1, use_reconstruction=True)

#    # Cross-modal retrieval
#    cross_modal_retrieval(test_loader=train_loaders[0], query_mod=1, gallery_mod=0, use_reconstruction=False)
#    cross_modal_retrieval(test_loader=train_loaders[0], query_mod=1, gallery_mod=0, use_reconstruction=True)







def encode_modality(node_idx, loader):
    embs = {node_idx: {node_idx:[]}}  # Only reconstruct for the given node_idx
    labels = {node_idx: []}

    for idx, modality in modality_node_dict.items():
        encoders[modality].eval()

    with torch.no_grad():
        for batch_num, batch in zip(tqdm(range(20)), loader):
            modalitiy_input = {mod: batch[mod].to(device) for mod in modalities}
            y = batch["label"].to(device)
            for mod_id, modality in modality_node_dict.items():
                h = encoders[modality](modalitiy_input[modality])  # [B, embed_dim]

                if mod_id == node_idx:
                    # Store direct embedding under key 0
                    embs[node_idx][node_idx].append(h)
                    if mod_id == node_idx:
                        labels[node_idx].append(y)
                else:
                    # Reconstruct node_idx embedding from mod_id
                    reconstructions = []
                    for (i, j) in edges:
                        if node_idx in (i, j) and mod_id in (i, j):
                            edge_key = f"{i}-{j}"
                            source_key = f"{mod_id}->{edge_key}"
                            dual_key = f"{edge_key}->{node_idx}"

                            P = restrictions[source_key]  # [edge_dim, embed_dim]
                            Q = duals[dual_key]           # [embed_dim, edge_dim]

                            edge_emb = h @ P.t()          # project to edge
                            recon = edge_emb @ Q.t()      # reconstruct at node_idx
                            reconstructions.append(recon)

                    if len(reconstructions) > 0:
                        recon_total = torch.stack(reconstructions).mean(dim=0)  # or sum
                        if mod_id not in embs[node_idx]:
                            embs[node_idx][mod_id] = []
                        embs[node_idx][mod_id].append(recon_total)

    

    # Concatenate all
    for key in embs[node_idx]:
        embs[node_idx][key] = torch.cat(embs[node_idx][key], dim=0)
    labels[node_idx] = torch.cat(labels[node_idx], dim=0)

    return embs, labels






import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap
import matplotlib.pyplot as plt




for node in modality_node_dict:

    embs, labels = encode_modality(node, test_loaders[0])

    all_embeddings = embs[node][node]
    all_labels = labels[node]

    X = all_embeddings.cpu().numpy()
    all_labels = all_labels.cpu().numpy()


    


    # Run t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
    X_2d = tsne.fit_transform(X)

    # Plotting: Color by digit class (0–9)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    legend = plt.legend(*scatter.legend_elements(), title="Digit Label", loc="best")
    plt.gca().add_artist(legend)
    plt.title(f"t-SNE of Multi-Modal Deepsense6G Embeddings at modality {modality_node_dict[node]} (Color-coded by Label)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

    plt.savefig(f"Tsne_node_{node}.png")








    for projected_node in modality_node_dict:

        if projected_node == node:
            continue
        # === Train UMAP on a reference set (direct embeddings) ===
        reference_embeddings = embs[node][projected_node]  # direct embeddings from node 0
        reference_labels = labels[node]

        X_ref = reference_embeddings.cpu().numpy()
        y_ref = reference_labels.cpu().numpy()

        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=0
        )
        X_ref_2d = umap_model.fit_transform(X_ref)

        # === Transform other embeddings (reconstructed from node 1) ===
        X_target = embs[node][1].cpu().numpy()
        y_target = labels[node].cpu().numpy()
        X_target_2d = umap_model.transform(X_target)

        # === Side-by-side plots ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot direct embeddings
        sc1 = axes[0].scatter(
            X_ref_2d[:, 0], X_ref_2d[:, 1],
            c=y_ref, cmap='tab10', alpha=0.8
        )
        axes[0].set_title(f"UMAP Projection of modality {modality_node_dict[node]} (Direct)")
        axes[0].set_xlabel("Dimension 1")
        axes[0].set_ylabel("Dimension 2")
        axes[0].grid(True)
        legend1 = axes[0].legend(*sc1.legend_elements(), title="Label", loc="best")
        axes[0].add_artist(legend1)

        # Plot reconstructed embeddings
        sc2 = axes[1].scatter(
            X_target_2d[:, 0], X_target_2d[:, 1],
            c=y_target, cmap='tab10', alpha=0.8
        )
        axes[1].set_title(f"UMAP Projection at modality {modality_node_dict[node]} reconstructed from modality {modality_node_dict[projected_node]}")
        axes[1].set_xlabel("Dimension 1")
        axes[1].set_ylabel("Dimension 2")
        axes[1].grid(True)
        legend2 = axes[1].legend(*sc2.legend_elements(), title="Label", loc="best")
        axes[1].add_artist(legend2)

        plt.tight_layout()

        plt.savefig(f"UMAP_node_{node}_projected_{projected_node}.png")
