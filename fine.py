import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec1 = nn.Conv2d(16, out_channels, 3, padding=1)
        
    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        e3 = F.relu(self.enc3(self.pool(e2)))
        
        d3 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.relu(self.dec3(d3 + e2))
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = F.relu(self.dec2(d2 + e1))
        out = self.dec1(d2)
        return out
