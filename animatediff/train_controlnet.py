# import sys
# import os
# import torch
# import torch.nn.functional as F_torch
# from torch.utils.data import DataLoader
# from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.utils as vutils
# import numpy as np
# import json
# from tqdm import tqdm

# sys.path.insert(0, '/data/Jayin/ltx/animatediff')
# from controlnet.dataset import DepthVideoDataset
# from controlnet.condition_encoder import ConditionEncoder
# from controlnet.controlnet_model import DepthControlNet

# # ── Config ───────────────────────────────────────────────────────────────────
# MODEL_PATH   = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
# DATA_DIR     = '/data/Jayin/ltx/data/processed'
# CKPT_DIR     = '/data/Jayin/ltx/animatediff/checkpoints'
# TB_DIR       = '/data/Jayin/ltx/animatediff/logs/tensorboard'

# BATCH_SIZE   = 1
# LR           = 1e-5
# MAX_STEPS    = 10000
# SAVE_EVERY   = 500
# LOG_EVERY    = 10
# VIS_EVERY    = 300      # visual validation every N steps
# NUM_WORKERS  = 4
# VAL_STEPS    = 20       # DDIM steps for validation inference

# os.makedirs(CKPT_DIR, exist_ok=True)
# os.makedirs(TB_DIR, exist_ok=True)

# # ── TensorBoard writer ────────────────────────────────────────────────────────
# writer = SummaryWriter(log_dir=TB_DIR)
# print(f"TensorBoard logs: {TB_DIR}")
# print(f"View with: tensorboard --logdir {TB_DIR} --port 6006")
# print()

# # ── Load pipeline ─────────────────────────────────────────────────────────────
# print("Loading AnimateDiff pipeline...")
# adapter = MotionAdapter.from_pretrained(
#     'guoyww/animatediff-motion-adapter-v1-5-2',
#     local_files_only=True
# )
# scheduler = DDIMScheduler.from_pretrained(
#     MODEL_PATH, subfolder='scheduler',
#     clip_sample=False, beta_schedule='linear',
#     timestep_spacing='linspace', steps_offset=1
# )
# pipe = AnimateDiffPipeline.from_pretrained(
#     MODEL_PATH, motion_adapter=adapter,
#     scheduler=scheduler, torch_dtype=torch.float16
# )
# pipe.to('cuda')

# unet      = pipe.unet
# vae       = pipe.vae
# text_enc  = pipe.text_encoder
# tokenizer = pipe.tokenizer

# unet.requires_grad_(False)
# vae.requires_grad_(False)
# text_enc.requires_grad_(False)

# VAE_SCALE = vae.config.scaling_factor
# print(f"Pipeline loaded — GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB used")

# # ── Build trainable modules ───────────────────────────────────────────────────
# print("\nBuilding ControlNet...")
# condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
# controlnet        = DepthControlNet(unet).cuda().float()

# n_train = (sum(p.numel() for p in controlnet.parameters()) +
#            sum(p.numel() for p in condition_encoder.parameters()))
# print(f"Trainable: {n_train/1e6:.0f}M params")
# print(f"GPU free:  {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

# # ── Dataset ───────────────────────────────────────────────────────────────────
# print("\nLoading dataset...")
# train_ds = DepthVideoDataset(DATA_DIR, split='train')
# val_ds   = DepthVideoDataset(DATA_DIR, split='val')
# train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
#                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
# val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
# print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")

# # ── Optimizer + scaler ────────────────────────────────────────────────────────
# optimizer = torch.optim.AdamW(
#     list(controlnet.parameters()) +
#     list(condition_encoder.parameters()),
#     lr=LR, weight_decay=1e-4
# )
# scaler = torch.cuda.amp.GradScaler()

# # ── Validation / visual logging ───────────────────────────────────────────────
# @torch.no_grad()
# def run_validation(step):
#     """
#     Generate conditioned vs unconditioned frames from a val batch.
#     Logs to TensorBoard:
#       - depth map (input conditioning)
#       - conditioned generation (model output with depth)
#       - unconditioned generation (model output without depth — zero condition)
#     """
#     controlnet.eval()
#     condition_encoder.eval()

#     # Get one validation batch
#     val_batch = next(iter(val_dl))
#     video = val_batch['video'].cuda()
#     depth = val_batch['depth'].cuda()
#     caption = val_batch['caption']

#     B, _, nf, H, W = video.shape

#     # ── Encode text ────────────────────────────────────────────────────────
#     tokens   = tokenizer(caption, padding='max_length', max_length=77,
#                          truncation=True, return_tensors='pt').input_ids.cuda()
#     text_emb = text_enc(tokens).last_hidden_state.float()
#     text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

#     # ── Start from random noise ────────────────────────────────────────────
#     latents = torch.randn(B, 4, nf, 64, 64, device='cuda', dtype=torch.float32)

#     # Set up scheduler for inference
#     scheduler.set_timesteps(VAL_STEPS)
#     timesteps = scheduler.timesteps

#     # ── Compute condition features once ────────────────────────────────────
#     depth_2d  = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)

#     with torch.cuda.amp.autocast():
#         cond_feat = condition_encoder(depth_2d)
#         cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

#     # Zero condition for unconditioned baseline
#     zero_cond = torch.zeros_like(cond_feat)

#     def denoise(latents_in, cond_features, use_conditioning):
#         """Run VAL_STEPS of DDIM denoising."""
#         lat = latents_in.clone()
#         for t in timesteps:
#             t_batch = t.unsqueeze(0).cuda()

#             with torch.cuda.amp.autocast():
#                 if use_conditioning:
#                     down_res, mid_res = controlnet(
#                         lat, cond_features, t_batch, text_exp
#                     )
#                     down_res_fp16 = tuple(r.half() for r in down_res)
#                     mid_res_fp16  = mid_res.half()
#                 else:
#                     # Zero residuals = no conditioning effect
#                     down_res_fp16 = tuple(
#                         torch.zeros_like(r.half())
#                         for r in controlnet(lat, zero_cond, t_batch, text_exp)[0]
#                     )
#                     mid_res_fp16 = torch.zeros(
#                         B*nf, 1280, 8, 8, device='cuda', dtype=torch.float16
#                     )

#                 noise_pred = unet(
#                     lat.half(), t_batch,
#                     encoder_hidden_states=text_exp.half(),
#                     down_block_additional_residuals=down_res_fp16,
#                     mid_block_additional_residual=mid_res_fp16,
#                 ).sample.float()

#             lat = scheduler.step(noise_pred, t, lat).prev_sample

#         return lat

#     # ── Run conditioned denoising ──────────────────────────────────────────
#     print(f"\n  [Step {step}] Running validation inference...")
#     lat_cond   = denoise(latents, cond_feat,  use_conditioning=True)
#     lat_uncond = denoise(latents, zero_cond,  use_conditioning=False)

#     # ── Decode latents → pixels ────────────────────────────────────────────
#     def decode(lat):
#         # lat: [B, 4, nf, 64, 64] → decode each frame
#         lat_2d = lat.permute(0,2,1,3,4).reshape(B*nf, 4, 64, 64).half()
#         frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
#         frames = (frames.clamp(-1, 1) + 1) / 2   # [0, 1]
#         return frames  # [B*nf, 3, 512, 512]

#     frames_cond   = decode(lat_cond)
#     frames_uncond = decode(lat_uncond)

#     # ── Prepare depth for visualization ────────────────────────────────────
#     # depth: [B, 1, nf, H, W] → [B*nf, 1, H, W] → repeat to 3ch
#     depth_vis = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
#     depth_vis = depth_vis.repeat(1, 3, 1, 1).float()  # [B*nf, 3, H, W]

#     # Resize depth to match generated frame size for side-by-side comparison
#     depth_vis = torch.nn.functional.interpolate(
#         depth_vis, size=(512, 512), mode='bilinear', align_corners=False
#     )

#     # ── Log to TensorBoard ─────────────────────────────────────────────────
#     # Make grids: each row is one frame, cols are [depth | cond | uncond]
#     comparison = torch.cat([depth_vis, frames_cond, frames_uncond], dim=0)
#     grid = vutils.make_grid(comparison, nrow=nf, normalize=False, padding=4)

#     writer.add_image('validation/depth_cond_uncond', grid, global_step=step)
#     writer.add_text('validation/caption', caption[0], global_step=step)

#     # Also log individual frame sets for easier inspection
#     writer.add_images('validation/depth_input',   depth_vis,    global_step=step)
#     writer.add_images('validation/conditioned',   frames_cond,  global_step=step)
#     writer.add_images('validation/unconditioned', frames_uncond, global_step=step)

#     print(f"  [Step {step}] Validation logged to TensorBoard")

#     controlnet.train()
#     condition_encoder.train()

# # ── Training loop ─────────────────────────────────────────────────────────────
# print(f"\nStarting training — {MAX_STEPS} steps")
# print(f"  LR: {LR} | Batch: {BATCH_SIZE} | Vis every: {VIS_EVERY} steps")
# print()

# controlnet.train()
# condition_encoder.train()

# step  = 0
# epoch = 0
# pbar  = tqdm(total=MAX_STEPS, desc="Training")

# while step < MAX_STEPS:
#     epoch += 1
#     for batch in train_dl:
#         if step >= MAX_STEPS:
#             break

#         video   = batch['video'].cuda()
#         depth   = batch['depth'].cuda()
#         caption = batch['caption']
#         B, _, nf, H, W = video.shape

#         # ── VAE encode ────────────────────────────────────────────────────
#         with torch.no_grad():
#             vid_2d  = video.half().permute(0,2,1,3,4).reshape(B*nf, 3, H, W)
#             latents = vae.encode(vid_2d).latent_dist.sample() * VAE_SCALE
#             latents = latents.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4).float()
#             noise   = torch.randn_like(latents)
#             t       = torch.randint(0, 1000, (B,), device='cuda', dtype=torch.long)
#             noisy   = scheduler.add_noise(latents, noise, t)
#             tokens  = tokenizer(caption, padding='max_length',
#                                 max_length=tokenizer.model_max_length,
#                                 truncation=True, return_tensors='pt').input_ids.cuda()
#             text_emb = text_enc(tokens).last_hidden_state.float()

#         text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)
#         depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)

#         # ── Forward ───────────────────────────────────────────────────────
#         with torch.cuda.amp.autocast():
#             cond_feat = condition_encoder(depth_2d)
#             cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

#             down_res, mid_res = controlnet(noisy, cond_feat, t, text_exp)

#             down_res_fp16 = tuple(r.half() for r in down_res)
#             mid_res_fp16  = mid_res.half()

#             noise_pred = unet(
#                 noisy.half(), t,
#                 encoder_hidden_states=text_exp.half(),
#                 down_block_additional_residuals=down_res_fp16,
#                 mid_block_additional_residual=mid_res_fp16,
#             ).sample.float()

#             loss = F_torch.mse_loss(noise_pred, noise)

#         # ── Backward ──────────────────────────────────────────────────────
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(
#             list(controlnet.parameters()) +
#             list(condition_encoder.parameters()),
#             max_norm=1.0
#         )
#         scaler.step(optimizer)
#         scaler.update()

#         step    += 1
#         loss_val = loss.item()

#         # ── Scalar logging ─────────────────────────────────────────────────
#         if step % LOG_EVERY == 0:
#             writer.add_scalar('train/loss', loss_val, step)
#             writer.add_scalar('train/grad_scale', scaler.get_scale(), step)
#             writer.add_scalar('train/gpu_gb', torch.cuda.memory_allocated()/1e9, step)

#         pbar.update(1)
#         pbar.set_postfix({
#             'loss':  f'{loss_val:.4f}',
#             'epoch': epoch,
#             'gpu':   f'{torch.cuda.memory_allocated()/1e9:.1f}G',
#             'scale': f'{scaler.get_scale():.0f}'
#         })

#         # ── Visual validation ──────────────────────────────────────────────
#         if step % VIS_EVERY == 0:
#             run_validation(step)

#         # ── Checkpoint ────────────────────────────────────────────────────
#         if step % SAVE_EVERY == 0:
#             ckpt_path = f'{CKPT_DIR}/step_{step:06d}.pt'
#             torch.save({
#                 'step':              step,
#                 'controlnet':        controlnet.state_dict(),
#                 'condition_encoder': condition_encoder.state_dict(),
#                 'optimizer':         optimizer.state_dict(),
#                 'scaler':            scaler.state_dict(),
#                 'loss':              loss_val,
#             }, ckpt_path)
#             tqdm.write(f'Saved: {ckpt_path}')

# pbar.close()
# writer.close()

# torch.save({
#     'step':              step,
#     'controlnet':        controlnet.state_dict(),
#     'condition_encoder': condition_encoder.state_dict(),
#     'optimizer':         optimizer.state_dict(),
# }, f'{CKPT_DIR}/final.pt')
# print(f"\nTraining complete — {step} steps")


























import sys
import os
import torch
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from PIL import Image
import json
from tqdm import tqdm

sys.path.insert(0, '/data/Jayin/ltx/animatediff')
from controlnet.dataset import DepthVideoDataset
from controlnet.condition_encoder import ConditionEncoder
from controlnet.controlnet_model import DepthControlNet

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
DATA_DIR    = '/data/Jayin/ltx/data/processed'
# CKPT_DIR    = '/data/Jayin/ltx/animatediff/checkpoints_clean'
# TB_DIR      = '/data/Jayin/ltx/animatediff/logs/tensorboard_clean'
# CLIP_LIST   = '/data/Jayin/ltx/animatediff/outdoor_clips_clean.json'
# RESUME_FROM = '/data/Jayin/ltx/animatediff/checkpoints_outdoor/step_010000.pt'
CKPT_DIR    = '/data/Jayin/ltx/animatediff/checkpoints_landscape'
TB_DIR      = '/data/Jayin/ltx/animatediff/logs/tensorboard_landscape'
CLIP_LIST   = '/data/Jayin/ltx/animatediff/landscape_clips.json'
RESUME_FROM = None          

BATCH_SIZE  = 1
# LR          = 1e-6
# MAX_STEPS   = 50000
# SAVE_EVERY  = 1000
LOG_EVERY   = 10
# VIS_EVERY   = 500
# VAL_STEPS   = 50
NUM_WORKERS = 4

LR          = 1e-5          # back to 1e-5 — fresh weights need higher LR
MAX_STEPS   = 20000
SAVE_EVERY  = 1000
VIS_EVERY   = 500
VAL_STEPS   = 50

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs('/data/Jayin/ltx/animatediff/logs', exist_ok=True)

# ── TensorBoard ───────────────────────────────────────────────────────────────
writer = SummaryWriter(log_dir=TB_DIR)
print(f"TensorBoard: {TB_DIR}")
print(f"Run: tensorboard --logdir {TB_DIR} --port 6006")
print()

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("Loading AnimateDiff pipeline...")
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
    MODEL_PATH,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16
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
print(f"Pipeline loaded — GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB used")
print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

# ── Build trainable modules ───────────────────────────────────────────────────
print("\nBuilding ControlNet...")
condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
controlnet        = DepthControlNet(unet).cuda().float()

n_train = (sum(p.numel() for p in controlnet.parameters()) +
           sum(p.numel() for p in condition_encoder.parameters()))
print(f"Trainable: {n_train/1e6:.0f}M params")

# ── Resume from checkpoint ────────────────────────────────────────────────────
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"\nResuming from: {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location='cuda')
    controlnet.load_state_dict(ckpt['controlnet'])
    condition_encoder.load_state_dict(ckpt['condition_encoder'])
    print(f"Loaded weights from step {ckpt['step']}")
    del ckpt
    torch.cuda.empty_cache()
else:
    print("\nStarting from scratch")

print(f"GPU used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

# ── Dataset ───────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
# with open(CLIP_LIST) as f:
#     outdoor_clips = json.load(f)
# print(f"Outdoor clip list: {len(outdoor_clips)} clips")

# train_ds = DepthVideoDataset(DATA_DIR, split='train', clip_list=outdoor_clips)
# val_ds   = DepthVideoDataset(DATA_DIR, split='val',   clip_list=outdoor_clips)

with open(CLIP_LIST) as f:
    landscape_clips = json.load(f)
print(f"Landscape clip list: {len(landscape_clips)} clips")

train_ds = DepthVideoDataset(DATA_DIR, split='train', clip_list=landscape_clips)
val_ds   = DepthVideoDataset(DATA_DIR, split='val',   clip_list=landscape_clips)



train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True
)
val_dl = DataLoader(
    val_ds, batch_size=1,
    shuffle=True, num_workers=2
)
print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")
print(f"Steps per epoch: {len(train_dl)}")

# ── Optimizer + scaler ────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(controlnet.parameters()) +
    list(condition_encoder.parameters()),
    lr=LR,
    weight_decay=1e-4
)
scaler = torch.cuda.amp.GradScaler()

# ── Validation function ───────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(step):
    controlnet.eval()
    condition_encoder.eval()

    # Get a random val batch — shuffle=True so different clip each time
    val_batch = next(iter(val_dl))
    video   = val_batch['video'].cuda()
    depth   = val_batch['depth'].cuda()
    caption = val_batch['caption']

    B, _, nf, H, W = video.shape

    # ── Encode text ────────────────────────────────────────────────────────
    tokens   = tokenizer(
        caption, padding='max_length', max_length=77,
        truncation=True, return_tensors='pt'
    ).input_ids.cuda()
    text_emb = text_enc(tokens).last_hidden_state.float()
    text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

    # ── Encode depth → condition features ─────────────────────────────────
    depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
    with torch.cuda.amp.autocast():
        cond_feat = condition_encoder(depth_2d)
        cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

    # ── Start from same random noise for fair comparison ───────────────────
    torch.manual_seed(step)  # reproducible per step
    latents = torch.randn(B, 4, nf, 64, 64, device='cuda', dtype=torch.float32)

    scheduler.set_timesteps(VAL_STEPS)
    timesteps = scheduler.timesteps

    def denoise(latents_in, use_conditioning):
        lat = latents_in.clone()
        for t in timesteps:
            t_batch = t.unsqueeze(0).cuda()
            with torch.cuda.amp.autocast():
                if use_conditioning:
                    down_res, mid_res = controlnet(lat, cond_feat, t_batch, text_exp)
                    noise_pred = unet(
                        lat.half(), t_batch,
                        encoder_hidden_states=text_exp.half(),
                        down_block_additional_residuals=tuple(r.half() for r in down_res),
                        mid_block_additional_residual=mid_res.half(),
                    ).sample.float()
                else:
                    # Proper unconditioned — no residuals at all
                    noise_pred = unet(
                        lat.half(), t_batch,
                        encoder_hidden_states=text_exp.half(),
                    ).sample.float()
            lat = scheduler.step(noise_pred, t, lat).prev_sample
        return lat

    tqdm.write(f"\n  [Step {step}] Running validation ({VAL_STEPS} DDIM steps)...")

    lat_cond   = denoise(latents, use_conditioning=True)
    lat_uncond = denoise(latents, use_conditioning=False)

    # ── Decode latents → frames ────────────────────────────────────────────
    def decode_latents(lat):
        lat_2d = lat.permute(0,2,1,3,4).reshape(B*nf, 4, 64, 64).half()
        frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
        return (frames.clamp(-1, 1) + 1) / 2  # [B*nf, 3, H, W] in [0,1]

    frames_cond   = decode_latents(lat_cond)
    frames_uncond = decode_latents(lat_uncond)

    # ── Prepare depth visualization ────────────────────────────────────────
    depth_vis = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
    depth_vis = depth_vis.repeat(1, 3, 1, 1).float()
    depth_vis = torch.nn.functional.interpolate(
        depth_vis, size=(512, 512), mode='bilinear', align_corners=False
    )

    # ── Log to TensorBoard ─────────────────────────────────────────────────
    writer.add_images('val/depth_input',   depth_vis,    global_step=step)
    writer.add_images('val/conditioned',   frames_cond,  global_step=step)
    writer.add_images('val/unconditioned', frames_uncond, global_step=step)
    writer.add_text('val/caption', caption[0], global_step=step)

    # Also save to disk for easy inspection
    out_dir = f'/data/Jayin/ltx/animatediff/logs/val_outdoor/step_{step:06d}'
    os.makedirs(out_dir, exist_ok=True)

    for i in range(nf):
        d = (depth_vis[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        c = (frames_cond[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        u = (frames_uncond[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')

        # Save side by side: depth | conditioned | unconditioned
        combined = Image.new('RGB', (512*3, 512))
        combined.paste(Image.fromarray(d), (0,     0))
        combined.paste(Image.fromarray(c), (512,   0))
        combined.paste(Image.fromarray(u), (1024,  0))
        combined.save(f'{out_dir}/frame_{i:02d}.png')

    tqdm.write(f"  [Step {step}] Saved to {out_dir}")

    controlnet.train()
    condition_encoder.train()

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting outdoor training")
print(f"  Max steps:  {MAX_STEPS}")
print(f"  LR:         {LR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Save every: {SAVE_EVERY}")
print(f"  Vis every:  {VIS_EVERY}")
print()

controlnet.train()
condition_encoder.train()

step  = 0
epoch = 0
pbar  = tqdm(total=MAX_STEPS, desc="Training")

while step < MAX_STEPS:
    epoch += 1

    for batch in train_dl:
        if step >= MAX_STEPS:
            break

        video   = batch['video'].cuda()
        depth   = batch['depth'].cuda()
        caption = batch['caption']
        B, _, nf, H, W = video.shape

        # ── VAE encode video ───────────────────────────────────────────────
        with torch.no_grad():
            vid_2d  = video.half().permute(0,2,1,3,4).reshape(B*nf, 3, H, W)
            latents = vae.encode(vid_2d).latent_dist.sample() * VAE_SCALE
            latents = latents.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4).float()
            noise   = torch.randn_like(latents)
            t       = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (B,), device='cuda', dtype=torch.long
            )
            noisy   = scheduler.add_noise(latents, noise, t)
            tokens  = tokenizer(
                caption, padding='max_length',
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors='pt'
            ).input_ids.cuda()
            text_emb = text_enc(tokens).last_hidden_state.float()

        # Expand text to B*nf
        text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

        # ── Condition encoder + ControlNet forward ─────────────────────────
        depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)

        with torch.cuda.amp.autocast():
            cond_feat = condition_encoder(depth_2d)
            cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

            down_res, mid_res = controlnet(noisy, cond_feat, t, text_exp)

            down_res_fp16 = tuple(r.half() for r in down_res)
            mid_res_fp16  = mid_res.half()

            noise_pred = unet(
                noisy.half(), t,
                encoder_hidden_states=text_exp.half(),
                down_block_additional_residuals=down_res_fp16,
                mid_block_additional_residual=mid_res_fp16,
            ).sample.float()

            loss = F_torch.mse_loss(noise_pred, noise)

        # ── Backward ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(controlnet.parameters()) +
            list(condition_encoder.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        step    += 1
        loss_val = loss.item()

        # ── Scalar logging ─────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            writer.add_scalar('train/loss',       loss_val,              step)
            writer.add_scalar('train/grad_scale', scaler.get_scale(),    step)
            writer.add_scalar('train/gpu_gb',     torch.cuda.memory_allocated()/1e9, step)
            writer.add_scalar('train/epoch',      epoch,                 step)

        pbar.update(1)
        pbar.set_postfix({
            'loss':  f'{loss_val:.4f}',
            'epoch': epoch,
            'gpu':   f'{torch.cuda.memory_allocated()/1e9:.1f}G',
            'scale': f'{scaler.get_scale():.0f}'
        })

        # ── Visual validation ──────────────────────────────────────────────
        if step % VIS_EVERY == 0:
            run_validation(step)

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            ckpt_path = f'{CKPT_DIR}/step_{step:06d}.pt'
            torch.save({
                'step':              step,
                'controlnet':        controlnet.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'optimizer':         optimizer.state_dict(),
                'scaler':            scaler.state_dict(),
                'loss':              loss_val,
            }, ckpt_path)
            tqdm.write(f'Saved: {ckpt_path}')

pbar.close()
writer.close()

# ── Final checkpoint ──────────────────────────────────────────────────────────
final_path = f'{CKPT_DIR}/final.pt'
torch.save({
    'step':              step,
    'controlnet':        controlnet.state_dict(),
    'condition_encoder': condition_encoder.state_dict(),
    'optimizer':         optimizer.state_dict(),
}, final_path)
print(f"\nTraining complete — {step} steps")
print(f"Final checkpoint: {final_path}")








