
from scipy.io import loadmat
import numpy as np
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import Dataset, dataloader, WeightedRandomSampler
from PIL import Image
import torch
import os
class FutureClearWindowDataset(Dataset):
    def __init__(self, csv_path, root_dir='.', window_length=16, T_f=5, transform=None):
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
            'mmwave': power,
            'RGB': rgb,
            'label': label
        }

