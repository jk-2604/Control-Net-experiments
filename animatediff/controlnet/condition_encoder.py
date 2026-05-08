import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Maps conditioning map from pixel space to latent space.
    Input:  [B*F, in_channels, 512, 512]
    Output: [B*F, 4, 64, 64]
    3x stride-2 convolutions = 8x spatial downsampling matching VAE.
    """

    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # 256
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 128
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),  # 64
        )

        # Zero-init final layer — outputs zero at init
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x):
        # x: [B*F, in_channels, 512, 512]
        return self.encoder(x)   # [B*F, 4, 64, 64]