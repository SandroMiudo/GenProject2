import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import ResidualBlock

# for now assuming fixed patch_sizes of 64
class Critic(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm((64, 64, 64)),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, (64, 64, 64), stride=1, norm_layer='layer'),
            ResidualBlock(64, 64, (64, 64, 64), stride=1, norm_layer='layer')
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, (128, 32, 32), stride=2, norm_layer='layer'),
            ResidualBlock(128, 128, (128, 32, 32), stride=1, norm_layer='layer')
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, (256, 16, 16), stride=2, norm_layer='layer'),
            ResidualBlock(256, 256, (256, 16, 16), stride=1, norm_layer='layer')
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, (512, 8, 8), stride=2, norm_layer='layer'),
            ResidualBlock(512, 512, (512, 8, 8), stride=1, norm_layer='layer')
        )

        self.layer5 = nn.Conv2d(512, 1, (8,8))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = torch.squeeze(x)

        return x