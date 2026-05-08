import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encoder-decoder for depth maps.
    Main output: latent-space features [B*F, 4, 64, 64]
    Aux output:  depth reconstruction [B*F, 1, 512, 512]
                 used only during training to force meaningful features
    """

    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),          nn.SiLU(),
        )  # [B, 32, 512, 512]

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.SiLU(),
        )  # [B, 64, 256, 256]

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),          nn.SiLU(),
        )  # [B, 128, 128, 128]

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),           nn.SiLU(),
        )  # [B, 256, 64, 64]

        # ── Latent projection ──────────────────────────────────────────────
        self.to_latent = nn.Conv2d(256, out_channels, 1)
        # [B, 4, 64, 64] — matches VAE latent space

        # ── Decoder for auxiliary reconstruction loss ──────────────────────
        # Used during training only — forces encoder to learn depth features
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
        )  # [B, 128, 128, 128]

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.SiLU(),
        )  # [B, 64, 256, 256]

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
        )  # [B, 32, 512, 512]

        self.to_depth = nn.Conv2d(32, in_channels, 1)
        # Reconstruction output [B, 1, 512, 512]

    def forward(self, x, return_reconstruction=False):
        """
        x: [B*F, in_channels, 512, 512]
        returns: [B*F, 4, 64, 64]
        if return_reconstruction=True: also returns [B*F, 1, 512, 512]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        latent = self.to_latent(e4)  # [B*F, 4, 64, 64]

        if not return_reconstruction:
            return latent

        # Decode for auxiliary loss
        d3   = self.dec3(e4)
        d2   = self.dec2(d3)
        d1   = self.dec1(d2)
        recon = torch.sigmoid(self.to_depth(d1))  # [B*F, 1, 512, 512] in [0,1]

        return latent, recon