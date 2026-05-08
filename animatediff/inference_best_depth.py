import sys
import os
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from PIL import Image, ImageDraw
import json

sys.path.insert(0, '/data/Jayin/ltx/animatediff')
from controlnet.dataset import DepthVideoDataset
from controlnet.condition_encoder import ConditionEncoder
from controlnet.controlnet_model import DepthControlNet

MODEL_PATH     = '/data/Jayin/ltx/animatediff/models/sd-v1-5'
DATA_DIR       = '/data/Jayin/ltx/data/processed'
CLIP_LIST      = '/data/Jayin/ltx/animatediff/outdoor_clips_clean.json'
CKPT_PATH      = '/data/Jayin/ltx/animatediff/checkpoints_clean/step_010000.pt'
OUT_DIR        = '/data/Jayin/ltx/animatediff/logs/best_depth_test'
GUIDANCE_SCALE = 7.5
DDIM_STEPS     = 50
SEED           = 42

# Best structured depth maps from val set
BEST_INDICES   = [56, 9, 27, 57, 22, 6]

# Fixed prompt — same for all
FIXED_PROMPT   = 'outdoor landscape, natural scenery, high quality video'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load everything ───────────────────────────────────────────────────────────
print("Loading pipeline...")
adapter = MotionAdapter.from_pretrained(
    'guoyww/animatediff-motion-adapter-v1-5-2', local_files_only=True)
scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH, subfolder='scheduler',
    clip_sample=False, beta_schedule='linear',
    timestep_spacing='linspace', steps_offset=1)
pipe = AnimateDiffPipeline.from_pretrained(
    MODEL_PATH, motion_adapter=adapter,
    scheduler=scheduler, torch_dtype=torch.float16)
pipe.to('cuda')

unet=pipe.unet; vae=pipe.vae; text_enc=pipe.text_encoder; tokenizer=pipe.tokenizer
unet.requires_grad_(False); vae.requires_grad_(False); text_enc.requires_grad_(False)
VAE_SCALE = vae.config.scaling_factor

condition_encoder = ConditionEncoder(in_channels=1, out_channels=4).cuda().float()
controlnet        = DepthControlNet(unet).cuda().float()
ckpt = torch.load(CKPT_PATH, map_location='cuda')
controlnet.load_state_dict(ckpt['controlnet'])
condition_encoder.load_state_dict(ckpt['condition_encoder'])
controlnet.eval(); condition_encoder.eval()
print(f"Loaded step {ckpt['step']}")

with open(CLIP_LIST) as f:
    clips = json.load(f)
val_ds = DepthVideoDataset(DATA_DIR, split='val', clip_list=clips)

# ── Text embeddings ───────────────────────────────────────────────────────────
with torch.no_grad():
    tokens = tokenizer([FIXED_PROMPT], padding='max_length', max_length=77,
                       truncation=True, return_tensors='pt').input_ids.cuda()
    text_emb = text_enc(tokens).last_hidden_state.float()

    empty_tokens = tokenizer([''], padding='max_length', max_length=77,
                             truncation=True, return_tensors='pt').input_ids.cuda()
    empty_text_emb = text_enc(empty_tokens).last_hidden_state.float()

# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run(depth, seed=42):
    B,_,nf,H,W = depth.shape
    text_exp  = text_emb.unsqueeze(1).expand(-1,nf,-1,-1).reshape(B*nf,77,768)
    empty_exp = empty_text_emb.unsqueeze(1).expand(-1,nf,-1,-1).reshape(B*nf,77,768)

    depth_2d = depth.permute(0,2,1,3,4).reshape(B*nf,1,H,W)
    with torch.cuda.amp.autocast():
        cond_feat = condition_encoder(depth_2d)
        cond_feat = cond_feat.reshape(B,nf,4,64,64).permute(0,2,1,3,4)

    torch.manual_seed(seed)
    latents = torch.randn(B,4,nf,64,64,device='cuda',dtype=torch.float32)
    scheduler.set_timesteps(DDIM_STEPS)

    for t in scheduler.timesteps:
        t_b = t.unsqueeze(0).cuda()
        with torch.cuda.amp.autocast():
            down_res, mid_res = controlnet(latents, cond_feat, t_b, text_exp)
            nc = unet(latents.half(), t_b,
                encoder_hidden_states=text_exp.half(),
                down_block_additional_residuals=tuple(r.half() for r in down_res),
                mid_block_additional_residual=mid_res.half()).sample.float()
            nu = unet(latents.half(), t_b,
                encoder_hidden_states=empty_exp.half()).sample.float()
            noise = nu + GUIDANCE_SCALE * (nc - nu)
        latents = scheduler.step(noise, t, latents).prev_sample

    lat_2d = latents.permute(0,2,1,3,4).reshape(B*nf,4,64,64).half()
    frames = vae.decode(lat_2d / VAE_SCALE).sample.float()
    return (frames.clamp(-1,1)+1)/2

# ── Build comparison grid ─────────────────────────────────────────────────────
# Each row: depth map | frame0 | frame1 | frame2 | frame3
print(f"\nRunning {len(BEST_INDICES)} clips — same prompt, same seed, different depths")
print(f"Prompt: {FIXED_PROMPT}")
print()

cell = 256
n    = len(BEST_INDICES)
grid = Image.new('RGB', (cell*5, cell*n + 30*n), (20,20,20))
draw = ImageDraw.Draw(grid)

# Column headers
for j, label in enumerate(['Depth', 'Frame 0', 'Frame 1', 'Frame 2', 'Frame 3']):
    draw.text((j*cell + cell//2 - 20, 5), label, fill=(200,200,200))

for row, idx in enumerate(BEST_INDICES):
    batch = val_ds[idx]
    depth = batch['depth'].unsqueeze(0).cuda()
    caption = batch['caption']
    B,_,nf,H,W = depth.shape

    print(f"Clip {row+1}/{n} (val_idx={idx}): {caption[:60]}")

    frames = run(depth, seed=SEED)

    y = 30 + row * (cell + 30)

    # Depth — show first frame
    dv = depth[0,:,0,:,:].repeat(3,1,1).float().cpu()
    dv = F.interpolate(dv.unsqueeze(0),(cell,cell),
                       mode='bilinear',align_corners=False).squeeze(0)
    d_img = Image.fromarray(
        (dv.permute(1,2,0).numpy()*255).astype('uint8')
    )
    grid.paste(d_img, (0, y))

    # Generated frames — crop watermark (bottom 8%)
    for fi in range(min(nf, 4)):
        g = (frames[fi].permute(1,2,0).cpu().numpy()*255).astype('uint8')
        g_img = Image.fromarray(g).crop((0,0,512,470)).resize((cell,cell))
        grid.paste(g_img, (cell*(fi+1), y))

    # Label
    draw.text((5, y - 22), f'{row+1}: std={batch["depth"].std():.3f} | {caption[:35]}',
              fill=(180,180,180))

    print(f"  Done")

grid.save(f'{OUT_DIR}/best_depth_comparison.png')
print(f"\nSaved: {OUT_DIR}/best_depth_comparison.png")
print(f"SCP: scp wtc12@<ip>:{OUT_DIR}/best_depth_comparison.png ~/Desktop/")