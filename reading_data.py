import pandas as pd
import os

# Parameters
csv_path = "Deepsense_dataset.csv"
output_dir = "dataset"
num_splits = 1
train_ratio = 0.7

# Load data
df = pd.read_csv(csv_path)
total_rows = len(df)

# Compute window and step sizes
window_size = total_rows // num_splits
step_size = window_size // 2  # 50% overlap

# Create subdirectories
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Generate and save train/test splits for each user
for user_index in range(num_splits):
    start_idx = user_index * step_size
    end_idx = start_idx + window_size
    if end_idx > total_rows:
        break  # Avoid short or incomplete segments

    split_df = df.iloc[start_idx:end_idx]

    train_end = int(len(split_df) * train_ratio)
    train_df = split_df.iloc[:train_end]
    test_df = split_df.iloc[train_end:]

    train_df.to_csv(f"{train_dir}/user_{user_index}.csv", index=False)
    test_df.to_csv(f"{test_dir}/user_{user_index}.csv", index=False)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision
from torchvision.transforms import transforms

import pandas as pd
import numpy as np
from scipy.io import loadmat
from PIL import Image
import os

import matplotlib.pyplot as plt




class FutureClearWindowDataset(Dataset):
    def __init__(self, csv_path, root_dir='.', window_length=4, T_f=2, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.window_length = window_length
        self.T_f = T_f
        self.transform = transform
        self.resize = transforms.Resize((64, 64))  # Resize to target dimensions


        # Total number of valid sliding windows
        self.valid_indices = [
            i for i in range(len(self.df) - window_length - T_f)
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_length
        future_start = end_idx
        future_end = future_start + self.T_f

        window_df = self.df.iloc[start_idx:end_idx]
        future_df = self.df.iloc[future_start:future_end]

        # Load data for window
        lidar_frames = []
        power_frames = []
        rgb_frames = []

        for row in window_df.itertuples():
            lidar = 0#loadmat(os.path.join(self.root_dir, row.unit1_lidar_SCR))['data']
            power = 0#np.loadtxt(os.path.join(self.root_dir, row.unit1_pwr_60ghz)[0:-4] + "_fixed.txt")
            rgb = 0#Image.open(os.path.join(self.root_dir, row.unit1_rgb)).convert('L')
            #rgb = self.resize(rgb)
            lidar_frames.append(lidar)
            power_frames.append(power)
            rgb_frames.append(rgb)
        lidar = 0#torch.tensor(np.stack(lidar_frames), dtype=torch.float32).permute(2, 0, 1)
        #lidar[0, :, :] = lidar[0, :, :] / 16.392
        #lidar[1, :, :] = (lidar[1] - (-2.0941)) / (1.5621 - (-2.0941))
        power = torch.tensor(np.stack(power_frames), dtype=torch.float32).unsqueeze(1)
        rgb = torch.tensor(np.stack(rgb_frames), dtype=torch.float32)

        # Future label logic: label = 1 if no blockage in future window
        future_blockages = future_df['blockage_label'].astype(float).values
        label = 1 - int(np.all(future_blockages == 0))  # 1 = clear, 0 = blockage ahead

        label = torch.tensor(label, dtype=torch.long)


        return {
            'lidar': lidar,      # shape (window_length, ...)
            'power': power,
            'rgb': rgb,
            'label': label
        }
    




    train_dir = "dataset/train"
output_dir = "dataset/weights"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(train_dir):
    if filename.endswith(".csv"):
        csv_path = os.path.join(train_dir, filename)
        print(f"Processing {filename}...")

        # Load dataset
        dataset = FutureClearWindowDataset(csv_path=csv_path, root_dir='.')

        # Compute labels
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1. / np.maximum(class_counts, 1)  # Avoid divide by zero
        print(set(labels))
        print(f"Weights: {class_weights}, count: {class_counts}")

        # Assign weights
        sample_weights = [class_weights[label] for label in labels]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

        # Save weights
        weight_path = os.path.join(output_dir, filename.replace('.csv', '_weights.pt'))
        torch.save(sample_weights_tensor, weight_path)



    

class FutureClearWindowDataset(Dataset):
    def __init__(self, csv_path, root_dir='.', window_length=4, T_f=2, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.window_length = window_length
        self.T_f = T_f
        self.transform = transform
        self.resize = transforms.Resize((64, 64))  # Resize to target dimensions


        # Total number of valid sliding windows
        self.valid_indices = [
            i for i in range(len(self.df) - window_length - T_f)
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_length
        future_start = end_idx
        future_end = future_start + self.T_f

        window_df = self.df.iloc[start_idx:end_idx]
        future_df = self.df.iloc[future_start:future_end]

        # Load data for window
        lidar_frames = []
        power_frames = []
        rgb_frames = []

        for row in window_df.itertuples():
            lidar = loadmat(os.path.join(self.root_dir, row.unit1_lidar_SCR))['data']
            power = np.loadtxt(os.path.join(self.root_dir, row.unit1_pwr_60ghz)[0:-4] + "_fixed.txt")
            rgb = Image.open(os.path.join(self.root_dir, row.unit1_rgb)).convert('L')
            rgb = self.resize(rgb) 
            lidar_frames.append(lidar)
            power_frames.append(power)
            rgb_frames.append(rgb)
        lidar = torch.tensor(np.stack(lidar_frames), dtype=torch.float32).permute(2, 0, 1)
        lidar[0, :, :] = lidar[0, :, :] / 16.392
        lidar[1, :, :] = (lidar[1] - (-2.0941)) / (1.5621 - (-2.0941))
        power = torch.tensor(np.stack(power_frames), dtype=torch.float32).unsqueeze(1)
        rgb = torch.tensor(np.stack(rgb_frames), dtype=torch.float32) / 255.0

        # Future label logic: label = 1 if no blockage in future window
        future_blockages = future_df['blockage_label'].astype(float).values
        label = 1 - int(np.all(future_blockages == 0))  # 1 = clear, 0 = blockage ahead

        label = torch.tensor(label, dtype=torch.long)


        return {
            'lidar': lidar,      # shape (window_length, ...)
            'power': power,
            'rgb': rgb,
            'label': label
        }
    




dataset = FutureClearWindowDataset(
    csv_path='dataset/train/user_0.csv',
    window_length=4,
    T_f=2
)

# Load from file
sample_weights_loaded = torch.load('dataset/weights/user_0_weights.pt', weights_only=True)


# Create sampler
sampler = WeightedRandomSampler(sample_weights_loaded, num_samples=len(sample_weights_loaded), replacement=True)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, sampler=sampler)


for batch in loader:
    print(batch['lidar'].shape, batch['power'].shape, batch['rgb'].shape, batch['label'])
    rgb_frames = batch['rgb']  # shape: (batch_size, num_frames, channels, height, width)

    # Let's plot the first image from the first batch (index 0)
    # We'll visualize a few frames from the first example in the batch

    num_frames_to_plot = 4  # Number of frames to visualize
    fig, axes = plt.subplots(1, num_frames_to_plot, figsize=(15, 5))
    print(batch["label"][0])
    for i in range(num_frames_to_plot):
        frame = rgb_frames[0, i, :, :]  # Get the first frame, with shape (height, width)
        print(frame)

        frame = frame.numpy()  # Convert tensor to numpy array for plotting
        axes[i].imshow(frame, cmap='gray')  # Display the image in grayscale
        axes[i].axis('off')  # Turn off axis labels
    
    plt.show()
    break