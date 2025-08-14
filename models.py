
import torch
import torch.nn as nn
import torch.functional as F

class SymmetricLiDARCNN(nn.Module):
    def __init__(self, input_channels=2, feature_dim=128):
        super(SymmetricLiDARCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # only reduce width
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # only reduce width
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # only reduce width
            nn.BatchNorm2d(128),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # (B, 128, 1, 1)
        x = self.flatten(x)        # (B, 128)
        x = self.fc(x)              # (B, feature_dim)
        return x



class RGBPredictionCNN(nn.Module):
    def __init__(self, input_channels=4, feature_dim=128, dropout_rate=0.01):
        super(RGBPredictionCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 64 → 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 32 → 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)  # 16 → 8

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 8 * 8, feature_dim)  # Outputs 128-D
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.flatten(x)  
        x = self.dropout(x)
        x = self.fc1(x)  # (B, feature_dim)
        return x

    



class mmWaveSCRNet(nn.Module):
    def __init__(self, input_channels=4, feature_dim=128, dropout_rate=0.2):
        super(mmWaveSCRNet, self).__init__()

        # Stack 1
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # W: 64 → 32

        # Stack 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))  # W: 32 → 16

        # Fusion Stack
        self.fusion_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_fusion1 = nn.BatchNorm2d(64)
        self.fusion_pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # W: 16 → 8

        self.fusion_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_fusion2 = nn.BatchNorm2d(64)
        self.fusion_pool2 = nn.MaxPool2d(kernel_size=(1, 2))  # W: 8 → 4

        # Fully Connected Projection to 128
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64 * 1 * 4, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv + Pool stages
        x = self.relu(self.bn1(self.conv1(x)))  
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))  
        x = self.pool2(x)

        x = self.relu(self.bn_fusion1(self.fusion_conv1(x)))  
        x = self.fusion_pool1(x)

        x = self.relu(self.bn_fusion2(self.fusion_conv2(x)))  
        x = self.fusion_pool2(x)

        # Flatten + FC
        x = torch.flatten(x, 1)  # (B, 64*1*4)
        x = self.dropout(x)
        x = self.fc(x)  # (B, 128)

        return x

