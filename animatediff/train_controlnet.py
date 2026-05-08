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





























# import sys
# import os
# import torch
# import torch.nn.functional as F_torch
# from torch.utils.data import DataLoader
# from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.utils as vutils
# from PIL import Image
# import json
# from tqdm import tqdm

# sys.path.insert(0, '/data/Jayin/ltx/animatediff')
# from controlnet.dataset import DepthVideoDataset
# from controlnet.condition_encoder import ConditionEncoder
# from controlnet.controlnet_model import DepthControlNet

# # ── Config ───────────────────────────────────────────────────────────────────
# MODEL_PATH  = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
# DATA_DIR    = '/data/Jayin/ltx/data/processed'
# # CKPT_DIR    = '/data/Jayin/ltx/animatediff/checkpoints_clean'
# # TB_DIR      = '/data/Jayin/ltx/animatediff/logs/tensorboard_clean'
# # CLIP_LIST   = '/data/Jayin/ltx/animatediff/outdoor_clips_clean.json'
# # RESUME_FROM = '/data/Jayin/ltx/animatediff/checkpoints_outdoor/step_010000.pt'
# CKPT_DIR    = '/data/Jayin/ltx/animatediff/checkpoints_landscape'
# TB_DIR      = '/data/Jayin/ltx/animatediff/logs/tensorboard_landscape'
# CLIP_LIST   = '/data/Jayin/ltx/animatediff/landscape_clips.json'
# RESUME_FROM = None          

# BATCH_SIZE  = 1
# # LR          = 1e-6
# # MAX_STEPS   = 50000
# # SAVE_EVERY  = 1000
# LOG_EVERY   = 10
# # VIS_EVERY   = 500
# # VAL_STEPS   = 50
# NUM_WORKERS = 4

# LR          = 1e-5          # back to 1e-5 — fresh weights need higher LR
# MAX_STEPS   = 20000
# SAVE_EVERY  = 1000
# VIS_EVERY   = 500
# VAL_STEPS   = 50

# os.makedirs(CKPT_DIR, exist_ok=True)
# os.makedirs(TB_DIR, exist_ok=True)
# os.makedirs('/data/Jayin/ltx/animatediff/logs', exist_ok=True)

# # ── TensorBoard ───────────────────────────────────────────────────────────────
# writer = SummaryWriter(log_dir=TB_DIR)
# print(f"TensorBoard: {TB_DIR}")
# print(f"Run: tensorboard --logdir {TB_DIR} --port 6006")
# print()

# # ── Load pipeline ─────────────────────────────────────────────────────────────
# print("Loading AnimateDiff pipeline...")
# adapter = MotionAdapter.from_pretrained(
#     'guoyww/animatediff-motion-adapter-v1-5-2',
#     local_files_only=True
# )
# scheduler = DDIMScheduler.from_pretrained(
#     MODEL_PATH, subfolder='scheduler',
#     clip_sample=False,
#     beta_schedule='linear',
#     timestep_spacing='linspace',
#     steps_offset=1
# )
# pipe = AnimateDiffPipeline.from_pretrained(
#     MODEL_PATH,
#     motion_adapter=adapter,
#     scheduler=scheduler,
#     torch_dtype=torch.float16
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
# print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

# # ── Build trainable modules ───────────────────────────────────────────────────
# print("\nBuilding ControlNet...")
# condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
# controlnet        = DepthControlNet(unet).cuda().float()

# n_train = (sum(p.numel() for p in controlnet.parameters()) +
#            sum(p.numel() for p in condition_encoder.parameters()))
# print(f"Trainable: {n_train/1e6:.0f}M params")

# # ── Resume from checkpoint ────────────────────────────────────────────────────
# if RESUME_FROM and os.path.exists(RESUME_FROM):
#     print(f"\nResuming from: {RESUME_FROM}")
#     ckpt = torch.load(RESUME_FROM, map_location='cuda')
#     controlnet.load_state_dict(ckpt['controlnet'])
#     condition_encoder.load_state_dict(ckpt['condition_encoder'])
#     print(f"Loaded weights from step {ckpt['step']}")
#     del ckpt
#     torch.cuda.empty_cache()
# else:
#     print("\nStarting from scratch")

# print(f"GPU used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
# print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

# # ── Dataset ───────────────────────────────────────────────────────────────────
# print("\nLoading dataset...")
# # with open(CLIP_LIST) as f:
# #     outdoor_clips = json.load(f)
# # print(f"Outdoor clip list: {len(outdoor_clips)} clips")

# # train_ds = DepthVideoDataset(DATA_DIR, split='train', clip_list=outdoor_clips)
# # val_ds   = DepthVideoDataset(DATA_DIR, split='val',   clip_list=outdoor_clips)

# with open(CLIP_LIST) as f:
#     landscape_clips = json.load(f)
# print(f"Landscape clip list: {len(landscape_clips)} clips")

# train_ds = DepthVideoDataset(DATA_DIR, split='train', clip_list=landscape_clips)
# val_ds   = DepthVideoDataset(DATA_DIR, split='val',   clip_list=landscape_clips)



# train_dl = DataLoader(
#     train_ds, batch_size=BATCH_SIZE,
#     shuffle=True, num_workers=NUM_WORKERS,
#     pin_memory=True, drop_last=True
# )
# val_dl = DataLoader(
#     val_ds, batch_size=1,
#     shuffle=True, num_workers=2
# )
# print(f"Train: {len(train_ds)} clips | Val: {len(val_ds)} clips")
# print(f"Steps per epoch: {len(train_dl)}")

# # ── Optimizer + scaler ────────────────────────────────────────────────────────
# optimizer = torch.optim.AdamW(
#     list(controlnet.parameters()) +
#     list(condition_encoder.parameters()),
#     lr=LR,
#     weight_decay=1e-4
# )
# scaler = torch.cuda.amp.GradScaler()

# # ── Validation function ───────────────────────────────────────────────────────
# @torch.no_grad()
# def run_validation(step):
#     controlnet.eval()
#     condition_encoder.eval()

#     # Get a random val batch — shuffle=True so different clip each time
#     val_batch = next(iter(val_dl))
#     video   = val_batch['video'].cuda()
#     depth   = val_batch['depth'].cuda()
#     caption = val_batch['caption']

#     B, _, nf, H, W = video.shape

#     # ── Encode text ────────────────────────────────────────────────────────
#     tokens   = tokenizer(
#         caption, padding='max_length', max_length=77,
#         truncation=True, return_tensors='pt'
#     ).input_ids.cuda()
#     text_emb = text_enc(tokens).last_hidden_state.float()
#     text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

#     # ── Encode depth → condition features ─────────────────────────────────
#     depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
#     with torch.cuda.amp.autocast():
#         cond_feat = condition_encoder(depth_2d)
#         cond_feat = cond_feat.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4)

#         if torch.rand(1).item() < 0.15:
#                 cond_feat = torch.zeros_like(cond_feat)

#     # ── Start from same random noise for fair comparison ───────────────────
#     torch.manual_seed(step)  # reproducible per step
#     latents = torch.randn(B, 4, nf, 64, 64, device='cuda', dtype=torch.float32)

#     scheduler.set_timesteps(VAL_STEPS)
#     timesteps = scheduler.timesteps

#     def denoise(latents_in, use_conditioning):
#         lat = latents_in.clone()
#         for t in timesteps:
#             t_batch = t.unsqueeze(0).cuda()
#             with torch.cuda.amp.autocast():
#                 if use_conditioning:
#                     down_res, mid_res = controlnet(lat, cond_feat, t_batch, text_exp)
#                     noise_pred = unet(
#                         lat.half(), t_batch,
#                         encoder_hidden_states=text_exp.half(),
#                         down_block_additional_residuals=tuple(r.half() for r in down_res),
#                         mid_block_additional_residual=mid_res.half(),
#                     ).sample.float()
#                 else:
#                     # Proper unconditioned — no residuals at all
#                     noise_pred = unet(
#                         lat.half(), t_batch,
#                         encoder_hidden_states=text_exp.half(),
#                     ).sample.float()
#             lat = scheduler.step(noise_pred, t, lat).prev_sample
#         return lat

#     tqdm.write(f"\n  [Step {step}] Running validation ({VAL_STEPS} DDIM steps)...")

#     lat_cond   = denoise(latents, use_conditioning=True)
#     lat_uncond = denoise(latents, use_conditioning=False)

#     # ── Decode latents → frames ────────────────────────────────────────────
#     def decode_latents(lat):
#         lat_2d = lat.permute(0,2,1,3,4).reshape(B*nf, 4, 64, 64).half()
#         frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
#         return (frames.clamp(-1, 1) + 1) / 2  # [B*nf, 3, H, W] in [0,1]

#     frames_cond   = decode_latents(lat_cond)
#     frames_uncond = decode_latents(lat_uncond)

#     # ── Prepare depth visualization ────────────────────────────────────────
#     depth_vis = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)
#     depth_vis = depth_vis.repeat(1, 3, 1, 1).float()
#     depth_vis = torch.nn.functional.interpolate(
#         depth_vis, size=(512, 512), mode='bilinear', align_corners=False
#     )

#     # ── Log to TensorBoard ─────────────────────────────────────────────────
#     writer.add_images('val/depth_input',   depth_vis,    global_step=step)
#     writer.add_images('val/conditioned',   frames_cond,  global_step=step)
#     writer.add_images('val/unconditioned', frames_uncond, global_step=step)
#     writer.add_text('val/caption', caption[0], global_step=step)

#     # Also save to disk for easy inspection
#     out_dir = f'/data/Jayin/ltx/animatediff/logs/val_outdoor/step_{step:06d}'
#     os.makedirs(out_dir, exist_ok=True)

#     for i in range(nf):
#         d = (depth_vis[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
#         c = (frames_cond[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
#         u = (frames_uncond[i].permute(1,2,0).cpu().numpy() * 255).astype('uint8')

#         # Save side by side: depth | conditioned | unconditioned
#         combined = Image.new('RGB', (512*3, 512))
#         combined.paste(Image.fromarray(d), (0,     0))
#         combined.paste(Image.fromarray(c), (512,   0))
#         combined.paste(Image.fromarray(u), (1024,  0))
#         combined.save(f'{out_dir}/frame_{i:02d}.png')

#     tqdm.write(f"  [Step {step}] Saved to {out_dir}")

#     controlnet.train()
#     condition_encoder.train()

# # ── Training loop ─────────────────────────────────────────────────────────────
# print(f"\nStarting outdoor training")
# print(f"  Max steps:  {MAX_STEPS}")
# print(f"  LR:         {LR}")
# print(f"  Batch size: {BATCH_SIZE}")
# print(f"  Save every: {SAVE_EVERY}")
# print(f"  Vis every:  {VIS_EVERY}")
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

#         # ── VAE encode video ───────────────────────────────────────────────
#         with torch.no_grad():
#             vid_2d  = video.half().permute(0,2,1,3,4).reshape(B*nf, 3, H, W)
#             latents = vae.encode(vid_2d).latent_dist.sample() * VAE_SCALE
#             latents = latents.reshape(B, nf, 4, 64, 64).permute(0,2,1,3,4).float()
#             noise   = torch.randn_like(latents)
#             t       = torch.randint(
#                 0, scheduler.config.num_train_timesteps,
#                 (B,), device='cuda', dtype=torch.long
#             )
#             noisy   = scheduler.add_noise(latents, noise, t)
#             tokens  = tokenizer(
#                 caption, padding='max_length',
#                 max_length=tokenizer.model_max_length,
#                 truncation=True, return_tensors='pt'
#             ).input_ids.cuda()
#             text_emb = text_enc(tokens).last_hidden_state.float()

#         # Expand text to B*nf
#         text_exp = text_emb.unsqueeze(1).expand(-1, nf, -1, -1).reshape(B*nf, 77, 768)

#         # ── Condition encoder + ControlNet forward ─────────────────────────
#         depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf, 1, H, W)

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
#             writer.add_scalar('train/loss',       loss_val,              step)
#             writer.add_scalar('train/grad_scale', scaler.get_scale(),    step)
#             writer.add_scalar('train/gpu_gb',     torch.cuda.memory_allocated()/1e9, step)
#             writer.add_scalar('train/epoch',      epoch,                 step)

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

# # ── Final checkpoint ──────────────────────────────────────────────────────────
# final_path = f'{CKPT_DIR}/final.pt'
# torch.save({
#     'step':              step,
#     'controlnet':        controlnet.state_dict(),
#     'condition_encoder': condition_encoder.state_dict(),
#     'optimizer':         optimizer.state_dict(),
# }, final_path)
# print(f"\nTraining complete — {step} steps")
# print(f"Final checkpoint: {final_path}")


















import sys
import os
import torch
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.utils as vutils
import numpy as np

# WandB — offline mode works without internet
os.environ['WANDB_MODE'] = 'offline'
import wandb

sys.path.insert(0, '/data/Jayin/ltx/animatediff')
from controlnet.dataset import DepthVideoDataset
from controlnet.condition_encoder import ConditionEncoder
from controlnet.controlnet_model import DepthControlNet

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
DATA_DIR    = '/data/Jayin/ltx/data/processed'
CKPT_DIR    = '/data/Jayin/ltx/animatediff/checkpoints_landscape_v2'
CLIP_LIST   = '/data/Jayin/ltx/animatediff/landscape_clips.json'
RESUME_FROM = None

BATCH_SIZE  = 1
LR          = 1e-5
MAX_STEPS   = 30000
SAVE_EVERY  = 1000
LOG_EVERY   = 10
VIS_EVERY   = 500
VAL_STEPS   = 30       # faster validation — 30 steps enough to detect collapse
NUM_WORKERS = 4
COND_DROP   = 0.15     # conditioning dropout rate

os.makedirs(CKPT_DIR, exist_ok=True)

# ── Init WandB ────────────────────────────────────────────────────────────────
wandb.init(
    project='animatediff_controlnet',
    name='landscape_v2_condrop',
    config={
        'lr': LR,
        'max_steps': MAX_STEPS,
        'batch_size': BATCH_SIZE,
        'cond_dropout': COND_DROP,
        'dataset': 'landscape_787clips',
        'clip_list': CLIP_LIST,
        'val_steps': VAL_STEPS,
    },
    dir='/data/Jayin/ltx/animatediff/logs/wandb'
)
os.makedirs('/data/Jayin/ltx/animatediff/logs/wandb', exist_ok=True)
print(f"WandB run: {wandb.run.name}  (offline mode)")

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("Loading AnimateDiff pipeline...")
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
print(f"Pipeline loaded — GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

# ── Build trainable modules ───────────────────────────────────────────────────
print("\nBuilding ControlNet...")
condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
controlnet        = DepthControlNet(unet).cuda().float()
n_train = sum(p.numel() for p in controlnet.parameters()) + \
          sum(p.numel() for p in condition_encoder.parameters())
print(f"Trainable: {n_train/1e6:.0f}M params")

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"Resuming from: {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location='cuda')
    controlnet.load_state_dict(ckpt['controlnet'])
    condition_encoder.load_state_dict(ckpt['condition_encoder'])
    print(f"Loaded step {ckpt['step']}")
    del ckpt; torch.cuda.empty_cache()

# ── Dataset ───────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
with open(CLIP_LIST) as f:
    landscape_clips = json.load(f)
print(f"Landscape clips: {len(landscape_clips)}")

train_ds = DepthVideoDataset(DATA_DIR, split='train', clip_list=landscape_clips)
val_ds   = DepthVideoDataset(DATA_DIR, split='val',   clip_list=landscape_clips)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=2)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Pre-select 4 fixed val clips for consistent visual tracking ───────────────
# Same 4 clips shown every validation — easy to track progress over time
# Pick clips with highest depth std from val set
val_depth_stats = []
for i in range(len(val_ds)):
    b = val_ds[i]
    val_depth_stats.append((i, float(b['depth'].std()), b['caption']))
val_depth_stats.sort(key=lambda x: x[1], reverse=True)
# Take top 4 with spread-out captions
FIXED_VAL_INDICES = [val_depth_stats[i][0] for i in [0, 1, 2, 3]]
print(f"\nFixed val clips for visual tracking:")
for idx in FIXED_VAL_INDICES:
    b = val_ds[idx]
    print(f"  idx={idx}  std={b['depth'].std():.3f}  {b['caption'][:60]}")

# ── Optimizer + scaler ────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(controlnet.parameters()) + list(condition_encoder.parameters()),
    lr=LR, weight_decay=1e-4
)
scaler = torch.cuda.amp.GradScaler()

# ── Validation function ───────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(step):
    controlnet.eval()
    condition_encoder.eval()

    # Fixed prompt — same every validation
    FIXED_PROMPT   = 'outdoor landscape, natural scenery'
    NEUTRAL_PROMPT = ''

    with torch.cuda.amp.autocast():
        tokens = tokenizer([FIXED_PROMPT], padding='max_length', max_length=77,
                           truncation=True, return_tensors='pt').input_ids.cuda()
        text_emb = text_enc(tokens).last_hidden_state.float()

        empty_tokens = tokenizer([NEUTRAL_PROMPT], padding='max_length', max_length=77,
                                 truncation=True, return_tensors='pt').input_ids.cuda()
        empty_emb = text_enc(empty_tokens).last_hidden_state.float()

    def denoise(depth, text_e, empty_e, seed=42):
        B,_,nf,H,W = depth.shape
        text_exp  = text_e.unsqueeze(1).expand(-1,nf,-1,-1).reshape(B*nf,77,768)
        empty_exp = empty_e.unsqueeze(1).expand(-1,nf,-1,-1).reshape(B*nf,77,768)

        depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf,1,H,W)
        with torch.cuda.amp.autocast():
            cf = condition_encoder(depth_2d)
            cf = cf.reshape(B,nf,4,64,64).permute(0,2,1,3,4)

        torch.manual_seed(seed)
        lat = torch.randn(B,4,nf,64,64,device='cuda',dtype=torch.float32)
        scheduler.set_timesteps(VAL_STEPS)

        for t in scheduler.timesteps:
            t_b = t.unsqueeze(0).cuda()
            with torch.cuda.amp.autocast():
                dr, mr = controlnet(lat, cf, t_b, text_exp)
                nc = unet(lat.half(), t_b,
                    encoder_hidden_states=text_exp.half(),
                    down_block_additional_residuals=tuple(r.half() for r in dr),
                    mid_block_additional_residual=mr.half()).sample.float()
                nu = unet(lat.half(), t_b,
                    encoder_hidden_states=empty_exp.half()).sample.float()
                noise = nu + 7.5*(nc - nu)
            lat = scheduler.step(noise, t, lat).prev_sample

        lat_2d = lat.permute(0,2,1,3,4).reshape(B*nf,4,64,64).half()
        frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
        return (frames.clamp(-1,1)+1)/2  # [nf, 3, H, W]

    # ── Build the collapse-detection grid ─────────────────────────────────────
    # Same structure as our fair_test — if all rows look identical = collapse
    # Each row: depth_map | frame0 | frame1 | frame2 | frame3
    cell = 200
    nf   = 4
    grid_img = Image.new('RGB', (cell*5, cell*4 + 40*4 + 30), (15,15,15))
    draw     = ImageDraw.Draw(grid_img)
    draw.text((5,5), f'Step {step} | {FIXED_PROMPT} | seed=42', fill=(220,220,220))

    depth_images   = []
    gen_images     = []
    residual_means = []

    for row, val_idx in enumerate(FIXED_VAL_INDICES):
        b     = val_ds[val_idx]
        depth = b['depth'].unsqueeze(0).cuda()
        cap   = b['caption']
        B,_,nf_d,H,W = depth.shape

        # Check residual magnitude — key collapse indicator
        depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf_d,1,H,W)
        t_test   = torch.tensor([500], dtype=torch.long).cuda()
        lat_test = torch.randn(B,4,nf_d,64,64,device='cuda')
        text_exp_test = text_emb.unsqueeze(1).expand(-1,nf_d,-1,-1).reshape(B*nf_d,77,768)
        with torch.cuda.amp.autocast():
            cf_test = condition_encoder(depth_2d)
            cf_test = cf_test.reshape(B,nf_d,4,64,64).permute(0,2,1,3,4)
            dr_test, mr_test = controlnet(lat_test, cf_test, t_test, text_exp_test)
        res_mean = float(mr_test.abs().mean())
        residual_means.append(res_mean)

        # Generate frames
        frames = denoise(depth, text_emb, empty_emb, seed=42)

        y = 30 + row*(cell+40)
        draw.text((5, y-22), f'std={b["depth"].std():.3f} | {cap[:45]}', fill=(160,160,160))

        # Depth
        dv = depth[0,:,0,:,:].repeat(3,1,1).float().cpu()
        dv = F_torch.interpolate(dv.unsqueeze(0),(cell,cell),
                                  mode='bilinear',align_corners=False).squeeze(0)
        d_img = Image.fromarray((dv.permute(1,2,0).numpy()*255).astype('uint8'))
        grid_img.paste(d_img, (0, y))
        depth_images.append(wandb.Image(d_img, caption=f'depth_{row+1}'))

        # Generated frames
        row_gen = []
        for fi in range(min(frames.shape[0], 4)):
            g = (frames[fi].permute(1,2,0).cpu().numpy()*255).astype('uint8')
            g_img = Image.fromarray(g).crop((0,0,512,470)).resize((cell,cell))
            grid_img.paste(g_img, (cell*(fi+1), y))
            row_gen.append(wandb.Image(g_img, caption=f'row{row+1}_frame{fi}'))
        gen_images.extend(row_gen)

    # ── Save grid image ────────────────────────────────────────────────────────
    grid_path = f'{CKPT_DIR}/val_step_{step:06d}.png'
    grid_img.save(grid_path)

    # ── Log everything to WandB ────────────────────────────────────────────────
    avg_res = float(np.mean(residual_means))

    wandb.log({
        # Collapse detection grid — 4 rows, different depth maps, same prompt/seed
        # If all rows look identical → collapse
        'val/collapse_detection_grid': wandb.Image(grid_img,
            caption=f'Step {step}: 4 depth maps, same prompt+seed. Identical=collapsed.'),

        # Individual depth maps
        'val/depth_map_row1': depth_images[0],
        'val/depth_map_row2': depth_images[1],
        'val/depth_map_row3': depth_images[2],
        'val/depth_map_row4': depth_images[3],

        # Individual generated frames (first frame of each clip)
        'val/gen_row1_frame0': gen_images[0],
        'val/gen_row2_frame0': gen_images[4],
        'val/gen_row3_frame0': gen_images[8],
        'val/gen_row4_frame0': gen_images[12],

        # Residual magnitude — should stay >0.1, drop to ~0 = collapse
        'val/mid_residual_mean': avg_res,
        'val/step': step,
    }, step=step)

    tqdm.write(f'\n  [Step {step}] Val: mid_res_mean={avg_res:.4f}  '
               f'{"⚠ COLLAPSING" if avg_res < 0.05 else "OK"}')
    tqdm.write(f'  Saved: {grid_path}')

    controlnet.train()
    condition_encoder.train()

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training — {MAX_STEPS} steps")
print(f"  LR: {LR} | Cond dropout: {COND_DROP} | Vis every: {VIS_EVERY}")
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
        B,_,nf,H,W = video.shape

        # VAE encode
        with torch.no_grad():
            vid_2d  = video.half().permute(0,2,1,3,4).reshape(B*nf,3,H,W)
            latents = vae.encode(vid_2d).latent_dist.sample() * VAE_SCALE
            latents = latents.reshape(B,nf,4,64,64).permute(0,2,1,3,4).float()
            noise   = torch.randn_like(latents)
            t       = torch.randint(0,scheduler.config.num_train_timesteps,
                                    (B,),device='cuda',dtype=torch.long)
            noisy   = scheduler.add_noise(latents, noise, t)
            tokens  = tokenizer(caption, padding='max_length',
                                max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt').input_ids.cuda()
            text_emb_train = text_enc(tokens).last_hidden_state.float()

        text_exp = text_emb_train.unsqueeze(1).expand(-1,nf,-1,-1).reshape(B*nf,77,768)
        depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf,1,H,W)

        with torch.cuda.amp.autocast():
            cond_feat = condition_encoder(depth_2d)
            cond_feat = cond_feat.reshape(B,nf,4,64,64).permute(0,2,1,3,4)

            # ── Conditioning dropout ───────────────────────────────────────
            # 15% of steps: zero condition → model learns unconditional path
            # Prevents mode collapse, enables proper CFG at inference
            if torch.rand(1).item() < COND_DROP:
                cond_feat = torch.zeros_like(cond_feat)

            down_res, mid_res = controlnet(noisy, cond_feat, t, text_exp)
            down_fp16 = tuple(r.half() for r in down_res)
            mid_fp16  = mid_res.half()

            noise_pred = unet(
                noisy.half(), t,
                encoder_hidden_states=text_exp.half(),
                down_block_additional_residuals=down_fp16,
                mid_block_additional_residual=mid_fp16,
            ).sample.float()

            loss = F_torch.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(controlnet.parameters()) + list(condition_encoder.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        step    += 1
        loss_val = loss.item()

        # Scalar logging
        if step % LOG_EVERY == 0:
            wandb.log({
                'train/loss':       loss_val,
                'train/grad_scale': scaler.get_scale(),
                'train/epoch':      epoch,
                'train/gpu_gb':     torch.cuda.memory_allocated()/1e9,
            }, step=step)

        pbar.update(1)
        pbar.set_postfix({
            'loss':  f'{loss_val:.4f}',
            'epoch': epoch,
            'gpu':   f'{torch.cuda.memory_allocated()/1e9:.1f}G',
            'scale': f'{scaler.get_scale():.0f}'
        })

        # Visual validation
        if step % VIS_EVERY == 0:
            run_validation(step)

        # Checkpoint
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
wandb.finish()

torch.save({
    'step': step,
    'controlnet': controlnet.state_dict(),
    'condition_encoder': condition_encoder.state_dict(),
    'optimizer': optimizer.state_dict(),
}, f'{CKPT_DIR}/final.pt')
print(f"\nTraining complete — {step} steps")



