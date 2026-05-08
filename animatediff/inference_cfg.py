import sys
import os
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from PIL import Image
import json
import numpy as np

sys.path.insert(0, '/data/Jayin/ltx/animatediff')
from controlnet.dataset import DepthVideoDataset
from controlnet.condition_encoder import ConditionEncoder
from controlnet.controlnet_model import DepthControlNet

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
DATA_DIR    = '/data/Jayin/ltx/data/processed'
CLIP_LIST   = '/data/Jayin/ltx/animatediff/outdoor_clips_clean.json'
OUT_DIR     = '/data/Jayin/ltx/animatediff/logs/cfg_inference'

# Try multiple checkpoints — use whichever had lowest loss
CKPT_PATH   = '/data/Jayin/ltx/animatediff/checkpoints_clean/step_050000.pt'

GUIDANCE_SCALE = 7.5   # CFG strength — higher = stronger conditioning
DDIM_STEPS     = 50    # inference steps
SEED           = 42
NUM_CLIPS      = 4     # how many val clips to run inference on

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("Loading pipeline...")
adapter = MotionAdapter.from_pretrained(
    'guoyww/animatediff-motion-adapter-v1-5-2',
    local_files_only=True
)
scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH, subfolder='scheduler',
    clip_sample=False,
    beta_schedule='linear',
    timestep_spacing='linspace',
    steps_offset=1
)
pipe = AnimateDiffPipeline.from_pretrained(
    MODEL_PATH, motion_adapter=adapter,
    scheduler=scheduler, torch_dtype=torch.float16
)
pipe.to('cuda')

unet      = pipe.unet
vae       = pipe.vae
text_enc  = pipe.text_encoder
tokenizer = pipe.tokenizer

unet.requires_grad_(False)
vae.requires_grad_(False)
text_enc.requires_grad_(False)

VAE_SCALE = vae.config.scaling_factor
print(f"Pipeline loaded — GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Load trained ControlNet ───────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CKPT_PATH}")
condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
controlnet        = DepthControlNet(unet).cuda().float()

ckpt = torch.load(CKPT_PATH, map_location='cuda')
controlnet.load_state_dict(ckpt['controlnet'])
condition_encoder.load_state_dict(ckpt['condition_encoder'])
controlnet.eval()
condition_encoder.eval()
print(f"Loaded step {ckpt['step']}  loss={ckpt.get('loss', 'N/A')}")

# ── Precompute empty text embedding for CFG ───────────────────────────────────
# Empty string = unconditional text direction
with torch.no_grad():
    empty_tokens = tokenizer(
        [""],
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    ).input_ids.cuda()
    empty_text_emb = text_enc(empty_tokens).last_hidden_state.float()
    # shape: [1, 77, 768]

# ── Dataset ───────────────────────────────────────────────────────────────────
with open(CLIP_LIST) as f:
    clips = json.load(f)

val_ds = DepthVideoDataset(DATA_DIR, split='val', clip_list=clips)
print(f"Val clips: {len(val_ds)}")

# ── CFG inference function ────────────────────────────────────────────────────
@torch.no_grad()
def run_cfg_inference(depth, text_exp, empty_text_exp,
                      guidance_scale=7.5, seed=42, ddim_steps=50):
    """
    Full CFG denoising loop.

    depth:          [B, 1, F, H, W]
    text_exp:       [B*F, 77, 768]  conditioned text
    empty_text_exp: [B*F, 77, 768]  empty text for CFG
    """
    B = depth.shape[0]
    nf = depth.shape[2]
    H = depth.shape[3]
    W = depth.shape[4]

    # ── Encode depth into condition features ───────────────────────────────
    depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
    with torch.cuda.amp.autocast():
        cond_feat = condition_encoder(depth_2d)         # [B*F, 4, 64, 64]
        cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

    # ── Start from random noise ────────────────────────────────────────────
    torch.manual_seed(seed)
    latents = torch.randn(B, 4, nf, 64, 64, device='cuda', dtype=torch.float32)

    scheduler.set_timesteps(ddim_steps)

    # ── Denoising loop with CFG ────────────────────────────────────────────
    for i, t in enumerate(scheduler.timesteps):
        t_batch = t.unsqueeze(0).cuda()

        with torch.cuda.amp.autocast():

            # ── Conditioned prediction ─────────────────────────────────────
            # ControlNet produces residuals from depth conditioning
            down_res_cond, mid_res_cond = controlnet(
                latents, cond_feat, t_batch, text_exp
            )
            # Cast residuals to fp16 for frozen UNet
            down_res_cond_fp16 = tuple(r.half() for r in down_res_cond)
            mid_res_cond_fp16  = mid_res_cond.half()

            noise_pred_cond = unet(
                latents.half(), t_batch,
                encoder_hidden_states=text_exp.half(),
                down_block_additional_residuals=down_res_cond_fp16,
                mid_block_additional_residual=mid_res_cond_fp16,
            ).sample.float()

            # ── Unconditioned prediction ───────────────────────────────────
            # No ControlNet residuals, empty text — pure unconditional
            noise_pred_uncond = unet(
                latents.half(), t_batch,
                encoder_hidden_states=empty_text_exp.half(),
            ).sample.float()

            # ── CFG combination ────────────────────────────────────────────
            # Push generation strongly toward the conditioned direction
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # Step the scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{ddim_steps}")

    return latents

# ── Run on multiple val clips ─────────────────────────────────────────────────
print(f"\nRunning CFG inference (guidance={GUIDANCE_SCALE}, steps={DDIM_STEPS})")
print(f"Output: {OUT_DIR}")
print()

for clip_idx in range(min(NUM_CLIPS, len(val_ds))):
    batch  = val_ds[clip_idx]
    video  = batch['video'].unsqueeze(0).cuda()   # [1, 3, F, H, W]
    depth  = batch['depth'].unsqueeze(0).cuda()   # [1, 1, F, H, W]
    caption = batch['caption']

    B, _, nf, H, W = video.shape
    print(f"Clip {clip_idx+1}/{NUM_CLIPS}: {caption[:70]}")

    # ── Encode text ────────────────────────────────────────────────────────
    with torch.no_grad():
        tokens   = tokenizer(
            [caption],
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.cuda()
        text_emb = text_enc(tokens).last_hidden_state.float()  # [1, 77, 768]

    # Expand to B*F
    text_exp       = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)
    empty_text_exp = empty_text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

    # ── Run inference ──────────────────────────────────────────────────────
    latents = run_cfg_inference(
        depth, text_exp, empty_text_exp,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        ddim_steps=DDIM_STEPS
    )

    # ── Decode latents ─────────────────────────────────────────────────────
    with torch.no_grad():
        lat_2d = latents.permute(0,2,1,3,4).reshape(B*nf, 4, 64, 64).half()
        frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
        frames = (frames.clamp(-1, 1) + 1) / 2   # [B*F, 3, 512, 512]

    # ── Prepare depth visualization ────────────────────────────────────────
    depth_vis = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
    depth_vis = depth_vis.repeat(1, 3, 1, 1).float()
    depth_vis = F.interpolate(depth_vis, (512, 512), mode='bilinear', align_corners=False)

    # ── Save one image per frame: depth (left) | generated (right) ────────
    clip_dir = f'{OUT_DIR}/clip_{clip_idx+1:02d}'
    os.makedirs(clip_dir, exist_ok=True)

    for f_idx in range(nf):
        d_np = (depth_vis[f_idx].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        g_np = (frames[f_idx].permute(1,2,0).cpu().numpy() * 255).astype('uint8')

        combined = Image.new('RGB', (1024, 512))
        combined.paste(Image.fromarray(d_np), (0,   0))
        combined.paste(Image.fromarray(g_np), (512, 0))
        combined.save(f'{clip_dir}/frame_{f_idx:02d}_depth_vs_gen.png')

    # Also save all 4 frames as a strip
    strip = Image.new('RGB', (512 * nf, 512))
    for f_idx in range(nf):
        g_np = (frames[f_idx].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        strip.paste(Image.fromarray(g_np), (512 * f_idx, 0))
    strip.save(f'{clip_dir}/all_frames_strip.png')

    print(f"  Saved to {clip_dir}")
    print()

print("Done. SCP to view:")
print(f"scp -r wtc12@<server_ip>:{OUT_DIR}/ ~/Desktop/cfg_inference/")