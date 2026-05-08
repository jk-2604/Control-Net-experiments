import sys
import os
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from PIL import Image, ImageDraw, ImageFont
import json

sys.path.insert(0, '/data/Jayin/ltx/animatediff')
from controlnet.dataset import DepthVideoDataset
from controlnet.condition_encoder import ConditionEncoder
from controlnet.controlnet_model import DepthControlNet

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
DATA_DIR       = '/data/Jayin/ltx/data/processed'
CLIP_LIST      = '/data/Jayin/ltx/animatediff/outdoor_clips_clean.json'
CKPT_PATH      = '/data/Jayin/ltx/animatediff/checkpoints_clean/step_050000.pt'
OUT_DIR        = '/data/Jayin/ltx/animatediff/logs/same_prompt_diff_depth'
GUIDANCE_SCALE = 7.5
DDIM_STEPS     = 50
SEED           = 42   # SAME seed for all — isolates depth effect

# Fixed prompt used for ALL clips
FIXED_PROMPT = "aerial view of green forest with winding path, natural landscape, high quality"

NUM_DEPTH_MAPS = 6   # number of different depth maps to try

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("Loading pipeline...")
adapter = MotionAdapter.from_pretrained(
    'guoyww/animatediff-motion-adapter-v1-5-2', local_files_only=True
)
scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH, subfolder='scheduler',
    clip_sample=False, beta_schedule='linear',
    timestep_spacing='linspace', steps_offset=1
)
pipe = AnimateDiffPipeline.from_pretrained(
    MODEL_PATH, motion_adapter=adapter,
    scheduler=scheduler, torch_dtype=torch.float16
)
pipe.to('cuda')

unet=pipe.unet; vae=pipe.vae; text_enc=pipe.text_encoder; tokenizer=pipe.tokenizer
unet.requires_grad_(False); vae.requires_grad_(False); text_enc.requires_grad_(False)
VAE_SCALE = vae.config.scaling_factor

# ── Load ControlNet ───────────────────────────────────────────────────────────
condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
controlnet        = DepthControlNet(unet).cuda().float()
ckpt = torch.load(CKPT_PATH, map_location='cuda')
controlnet.load_state_dict(ckpt['controlnet'])
condition_encoder.load_state_dict(ckpt['condition_encoder'])
controlnet.eval(); condition_encoder.eval()
print(f"Loaded step {ckpt['step']}")

# ── Precompute text embeddings (same for ALL clips) ────────────────────────────
print(f"\nFixed prompt: {FIXED_PROMPT}")
with torch.no_grad():
    # Conditioned text
    tokens = tokenizer(
        [FIXED_PROMPT],
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    ).input_ids.cuda()
    text_emb = text_enc(tokens).last_hidden_state.float()  # [1, 77, 768]

    # Empty text for CFG
    empty_tokens = tokenizer(
        [""],
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    ).input_ids.cuda()
    empty_text_emb = text_enc(empty_tokens).last_hidden_state.float()

# ── Load dataset — only for depth maps ────────────────────────────────────────
with open(CLIP_LIST) as f:
    clips = json.load(f)
val_ds = DepthVideoDataset(DATA_DIR, split='val', clip_list=clips)
print(f"Val clips available: {len(val_ds)}")

# ── CFG inference ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_cfg_inference(depth, text_exp, empty_text_exp,
                      guidance_scale=7.5, seed=42, ddim_steps=50):
    B   = depth.shape[0]
    nf  = depth.shape[2]
    H   = depth.shape[3]
    W   = depth.shape[4]

    depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
    with torch.cuda.amp.autocast():
        cond_feat = condition_encoder(depth_2d)
        cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

    torch.manual_seed(seed)
    latents = torch.randn(B, 4, nf, 64, 64, device='cuda', dtype=torch.float32)
    scheduler.set_timesteps(ddim_steps)

    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0).cuda()
        with torch.cuda.amp.autocast():
            down_res, mid_res = controlnet(latents, cond_feat, t_batch, text_exp)
            noise_pred_cond = unet(
                latents.half(), t_batch,
                encoder_hidden_states=text_exp.half(),
                down_block_additional_residuals=tuple(r.half() for r in down_res),
                mid_block_additional_residual=mid_res.half(),
            ).sample.float()

            noise_pred_uncond = unet(
                latents.half(), t_batch,
                encoder_hidden_states=empty_text_exp.half(),
            ).sample.float()

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents

@torch.no_grad()
def decode(latents, B, nf):
    lat_2d = latents.permute(0,2,1,3,4).reshape(B*nf, 4, 64, 64).half()
    frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
    return (frames.clamp(-1,1) + 1) / 2

# ── Run same prompt through N different depth maps ────────────────────────────
print(f"\nRunning {NUM_DEPTH_MAPS} clips with SAME prompt, SAME seed, DIFFERENT depths")
print(f"Seed: {SEED} | Guidance: {GUIDANCE_SCALE} | Steps: {DDIM_STEPS}")
print()

# Collect results for comparison grid
all_depth_frames   = []   # first frame depth vis per clip
all_gen_frames     = []   # first frame generated per clip
all_captions       = []   # original clip caption (for reference)

# Sample evenly from val set
indices = list(range(0, len(val_ds), max(1, len(val_ds)//NUM_DEPTH_MAPS)))[:NUM_DEPTH_MAPS]

for i, clip_idx in enumerate(indices):
    batch   = val_ds[clip_idx]
    depth   = batch['depth'].unsqueeze(0).cuda()    # [1, 1, F, H, W]
    caption = batch['caption']

    B, _, nf, H, W = depth.shape

    print(f"Clip {i+1}/{NUM_DEPTH_MAPS} (idx={clip_idx})")
    print(f"  Original caption: {caption[:60]}")
    print(f"  Using fixed prompt: {FIXED_PROMPT[:60]}")

    # Expand fixed text to B*F
    text_exp       = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)
    empty_text_exp = empty_text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

    # Run inference
    latents = run_cfg_inference(
        depth, text_exp, empty_text_exp,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,          # SAME seed every time
        ddim_steps=DDIM_STEPS
    )
    frames = decode(latents, B, nf)

    # Depth visualization (first frame only)
    depth_vis = depth[0, :, 0, :, :].repeat(3, 1, 1).float()   # [3, H, W]
    depth_vis = F.interpolate(
        depth_vis.unsqueeze(0), (512, 512), mode='bilinear', align_corners=False
    ).squeeze(0)

    # Collect first frame
    all_depth_frames.append(depth_vis.cpu())
    all_gen_frames.append(frames[0].cpu())   # first frame of sequence
    all_captions.append(caption[:40])

    # Save individual clip — all 4 frames side by side
    clip_dir = f'{OUT_DIR}/clip_{i+1:02d}'
    os.makedirs(clip_dir, exist_ok=True)

    # Save depth + all generated frames
    w_total = 512 * (nf + 1)
    strip = Image.new('RGB', (w_total, 512))

    # First panel: depth map
    d_np = (depth_vis.cpu().permute(1,2,0).numpy() * 255).astype('uint8')
    strip.paste(Image.fromarray(d_np), (0, 0))

    # Remaining panels: generated frames
    for f_idx in range(nf):
        g_np = (frames[f_idx].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        # Crop bottom 8% to remove watermark
        g_img = Image.fromarray(g_np)
        g_img = g_img.crop((0, 0, 512, 470))
        g_img = g_img.resize((512, 512))
        strip.paste(g_img, (512 * (f_idx + 1), 0))

    strip.save(f'{clip_dir}/depth_and_4frames.png')
    print(f"  Saved: {clip_dir}/depth_and_4frames.png")
    print()

# ── Build comparison grid ──────────────────────────────────────────────────────
# Grid: each row = one depth map + its generated first frame
# Shows clearly: different depth → different spatial layout
print("Building comparison grid...")

cell_w, cell_h = 512, 512
cols = 2   # depth | generated
rows = NUM_DEPTH_MAPS

grid = Image.new('RGB', (cell_w * cols + 20, cell_h * rows + rows * 30), color=(20, 20, 20))

for i in range(NUM_DEPTH_MAPS):
    y_offset = i * (cell_h + 30)

    # Depth
    d_np = (all_depth_frames[i].cpu().permute(1,2,0).numpy() * 255).astype('uint8')
    grid.paste(Image.fromarray(d_np), (0, y_offset + 30))

    # Generated — crop watermark
    g_np = (all_gen_frames[i].cpu().permute(1,2,0).numpy() * 255).astype('uint8')
    g_img = Image.fromarray(g_np).crop((0, 0, 512, 470)).resize((512, 512))
    grid.paste(g_img, (cell_w + 20, y_offset + 30))

    # Label
    draw = ImageDraw.Draw(grid)
    draw.text((5, y_offset + 8), f"Clip {i+1}: {all_captions[i]}", fill=(200, 200, 200))

grid.save(f'{OUT_DIR}/comparison_grid.png')
print(f"Saved comparison grid: {OUT_DIR}/comparison_grid.png")
print()
print("SCP to view:")
print(f"scp -r wtc12@<server_ip>:{OUT_DIR}/ ~/Desktop/same_prompt_diff_depth/")