import torch
import torch.nn as nn
import copy


class DepthControlNet(nn.Module):
    """
    ControlNet adapter for AnimateDiff UNetMotionModel.

    Produces exactly 12 down residuals + 1 mid residual matching
    the frozen UNet's internal skip connection structure:
      [0]        conv_in output:   (B*F, 320,  64, 64)
      [1,2,3]    block 0:          (B*F, 320,  64, 64) x2, (B*F, 320, 32, 32)
      [4,5,6]    block 1:          (B*F, 640,  32, 32) x2, (B*F, 640, 16, 16)
      [7,8,9]    block 2:          (B*F, 1280, 16, 16) x2, (B*F, 1280, 8, 8)
      [10,11]    block 3:          (B*F, 1280, 8,  8)  x2
      mid:                         (B*F, 1280, 8,  8)
    """

    def __init__(self, unet):
        super().__init__()

        # ── Time embedding ─────────────────────────────────────────────────
        self.time_proj      = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)

        # ── Encoder blocks ─────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList([
            copy.deepcopy(block) for block in unet.down_blocks
        ])
        self.mid_block = copy.deepcopy(unet.mid_block)

        # Enable gradient checkpointing on copied blocks
        # Recomputes activations during backward — saves ~40% activation memory
        for block in self.down_blocks:
            if hasattr(block, 'enable_gradient_checkpointing'):
                block.enable_gradient_checkpointing()
        if hasattr(self.mid_block, 'enable_gradient_checkpointing'):
            self.mid_block.enable_gradient_checkpointing()

        # ── Expanded conv_in: 4→320 becomes 8→320 ─────────────────────────
        # First 4 channels = noisy latent (copy original weights)
        # Last  4 channels = condition features (zero init)
        self.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv_in.weight[:, :4, :, :] = copy.deepcopy(unet.conv_in.weight)
            self.conv_in.weight[:, 4:, :, :] = 0.0
            self.conv_in.bias.copy_(unet.conv_in.bias)

        # ── Zero-convs: 12 for down + 1 for mid ───────────────────────────
        skip_channels = [
            320,                     # [0]    conv_in output
            320, 320, 320,           # [1-3]  block 0
            640, 640, 640,           # [4-6]  block 1
            1280, 1280, 1280,        # [7-9]  block 2
            1280, 1280,              # [10-11] block 3
        ]
        self.zero_convs    = nn.ModuleList([
            self._zero_conv(ch) for ch in skip_channels
        ])
        self.mid_zero_conv = self._zero_conv(1280)

        n = sum(p.numel() for p in self.parameters())
        print(f"DepthControlNet ready — {n/1e6:.0f}M params")
        print(f"  Zero-convs: {len(self.zero_convs)} down + 1 mid")

    def _zero_conv(self, channels):
        conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, noisy_latents, condition_features,
                timestep, encoder_hidden_states):
        """
        noisy_latents:         [B, 4, F, H, W]   float32
        condition_features:    [B, 4, F, H, W]   float32
        timestep:              [B]                long
        encoder_hidden_states: [B*F, 77, 768]     float32, pre-expanded

        Returns:
            all_skip_residuals: list of 12 tensors
            mid_residual:       tensor [B*F, 1280, 8, 8]
        """
        B, _, nf, H, W = noisy_latents.shape

        # ── Merge F into batch for 2D spatial ops ─────────────────────────
        lat  = noisy_latents.permute(0,2,1,3,4).reshape(B*nf, 4, H, W)
        cond = condition_features.permute(0,2,1,3,4).reshape(B*nf, 4, H, W)
        x    = torch.cat([lat, cond], dim=1)      # [B*F, 8, H, W]

        # ── conv_in ────────────────────────────────────────────────────────
        x = self.conv_in(x)                        # [B*F, 320, H, W]

        # ── First residual: conv_in output ─────────────────────────────────
        all_skip_residuals = [self.zero_convs[0](x)]
        zero_conv_idx      = 1

        # ── Time embedding ─────────────────────────────────────────────────
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=noisy_latents.dtype)
        t_emb = self.time_embedding(t_emb)          # [B, 1280]
        t_emb = t_emb.repeat_interleave(nf, dim=0)  # [B*F, 1280]

        text = encoder_hidden_states               # [B*F, 77, 768]

        # ── Down blocks ────────────────────────────────────────────────────
        for block in self.down_blocks:
            if hasattr(block, 'has_cross_attention') and block.has_cross_attention:
                x, res_samples = block(
                    hidden_states=x,
                    temb=t_emb,
                    encoder_hidden_states=text,
                    num_frames=nf
                )
            else:
                x, res_samples = block(
                    hidden_states=x,
                    temb=t_emb,
                    num_frames=nf
                )
            for res in res_samples:
                all_skip_residuals.append(
                    self.zero_convs[zero_conv_idx](res)
                )
                zero_conv_idx += 1

        # ── Mid block ──────────────────────────────────────────────────────
        x = self.mid_block(
            hidden_states=x,
            temb=t_emb,
            encoder_hidden_states=text,
            num_frames=nf
        )
        mid_residual = self.mid_zero_conv(x)

        return all_skip_residuals, mid_residual