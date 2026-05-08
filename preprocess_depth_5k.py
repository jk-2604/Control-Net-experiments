import torch
import numpy as np
from decord import VideoReader
from PIL import Image
import torchvision.transforms as T
import os
import json
from tqdm import tqdm
import time

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load MiDaS once ────────────────────────────────────────────────────────
print("\nLoading MiDaS DPT-Large...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval().to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform  = midas_transforms.dpt_transform
print("MiDaS ready")

# ── Video transform ────────────────────────────────────────────────────────
video_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # → [-1, 1]
])

# ── Find clips ─────────────────────────────────────────────────────────────
clip_dir = "data/raw_clips"
out_dir  = "data/processed"
os.makedirs(out_dir, exist_ok=True)

# Only process clips that were successfully downloaded
with open("download_results.json") as f:
    results = json.load(f)
succeeded_ids = set(results["succeeded"])

mp4_files = sorted([
    f for f in os.listdir(clip_dir)
    if f.endswith(".mp4")
    and f.replace(".mp4","") in succeeded_ids
])

# Skip already processed
already = set(os.listdir(out_dir))
mp4_files = [f for f in mp4_files if f.replace(".mp4","") not in already]

print(f"\nClips to process: {len(mp4_files)}")
print(f"Already processed: {len(already)}")

# ── Process ────────────────────────────────────────────────────────────────
processed = []
skipped   = []
start     = time.time()

for i, mp4_name in enumerate(tqdm(mp4_files, desc="Preprocessing")):
    clip_id  = mp4_name.replace(".mp4", "")
    mp4_path = f"{clip_dir}/{mp4_name}"
    save_dir = f"{out_dir}/{clip_id}"

    if os.path.exists(f"{save_dir}/depth.pt"):
        processed.append(clip_id)
        continue

    try:
        # ── Extract 8 frames ───────────────────────────────────────────────
        vr    = VideoReader(mp4_path)
        total = len(vr)

        if total < 8:
            skipped.append(clip_id)
            continue

        indices = np.linspace(0, total - 1, 8, dtype=int)
        frames  = vr.get_batch(indices).asnumpy()  # [8, H, W, 3]

        # ── Build tensors ──────────────────────────────────────────────────
        video_tensors = []
        raw_frames    = []

        for j in range(8):
            pil = Image.fromarray(frames[j]).resize((512, 512), Image.BILINEAR)
            raw_frames.append(np.array(pil))
            video_tensors.append(video_transform(pil))

        video_tensor = torch.stack(video_tensors)  # [8, 3, 512, 512] float32

        # ── MiDaS on all 8 frames ──────────────────────────────────────────
        depth_frames = []

        for raw in raw_frames:
            inp = midas_transform(raw).to(device)

            with torch.no_grad():
                pred = midas(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(512, 512),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_frames.append(pred.cpu())

        depth_tensor = torch.stack(depth_frames).unsqueeze(1)
        # [8, 1, 512, 512]

        # ── Clip-level normalize ───────────────────────────────────────────
        d_min      = depth_tensor.min()
        d_max      = depth_tensor.max()
        depth_norm = (depth_tensor - d_min) / (d_max - d_min + 1e-8)

        # ── Save as float16 to halve disk usage ────────────────────────────
        os.makedirs(save_dir, exist_ok=True)
        torch.save(video_tensor.half(), f"{save_dir}/video.pt")
        torch.save(depth_norm.half(),   f"{save_dir}/depth.pt")

        caption_path = f"{clip_dir}/{clip_id}.txt"
        caption = open(caption_path).read().strip() \
                  if os.path.exists(caption_path) else ""

        with open(f"{save_dir}/metadata.json", "w") as f:
            json.dump({
                "clip_id":       clip_id,
                "caption":       caption,
                "depth_min_raw": float(d_min),
                "depth_max_raw": float(d_max),
                "num_frames":    8
            }, f)

        processed.append(clip_id)

        # ── Progress report every 100 clips ───────────────────────────────
        if (i + 1) % 100 == 0:
            elapsed  = time.time() - start
            rate     = (i + 1) / elapsed * 60  # clips per minute
            remaining = (len(mp4_files) - i - 1) / (rate / 60)
            print(f"\n[{i+1}/{len(mp4_files)}] "
                  f"Rate: {rate:.1f} clips/min | "
                  f"ETA: {remaining/3600:.1f} hours | "
                  f"Disk: {get_disk_usage()}")

    except Exception as e:
        print(f"\nError {clip_id}: {e}")
        skipped.append(clip_id)

def get_disk_usage():
    import shutil
    total, used, free = shutil.disk_usage("/data/Jayin")
    return f"{free/1e9:.0f}GB free"

print(f"\nDone — processed: {len(processed)}, skipped: {len(skipped)}")

# Save final summary
with open("preprocess_summary.json", "w") as f:
    json.dump({
        "processed": processed,
        "skipped":   skipped
    }, f, indent=2)
