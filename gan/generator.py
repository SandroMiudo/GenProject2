import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
    UpBlock, 
    ResidualBlock
)

# for now assuming fixed patch_sizes of 64
class Generator(nn.Module):
    def __init__(self, out_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            UpBlock(512, 256, stride=2),
            ResidualBlock(256, 256, 0, stride=1)
        )
        self.layer2 = nn.Sequential(
            UpBlock(256, 128, stride=2),
            ResidualBlock(128, 128, 0, stride=1)
        )
        self.layer3 = nn.Sequential(
            UpBlock(128, 64, stride=2),
            ResidualBlock(64, 64, 0, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64, 0, stride=1),
            ResidualBlock(64, 64, 0, stride=1)
        )

        self.layer5 = nn.Conv2d(64, out_channels, (1,1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x
    
    @torch.no_grad
    def sample(self, n, x=None):
        z = torch.randn(n, self.latent_dim, 1, 1).cuda()
        return self.forward(z)