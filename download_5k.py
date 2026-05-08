import requests
import csv
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

os.makedirs("data/raw_clips", exist_ok=True)

# Load metadata
with open("webvid_5k.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total clips to download: {len(rows)}")

# Check what's already done
already_done = set(
    f.replace(".mp4", "")
    for f in os.listdir("data/raw_clips")
    if f.endswith(".mp4") and os.path.getsize(f"data/raw_clips/{f}") > 10000
)
print(f"Already downloaded: {len(already_done)}")

todo = [r for r in rows if r['videoid'] not in already_done]
print(f"Remaining: {len(todo)}")

# Thread-safe counters
lock = threading.Lock()
succeeded = list(already_done)
failed = []

def download_one(row):
    clip_id = row['videoid']
    url     = row['contentUrl']
    caption = row['caption']

    out_video   = f"data/raw_clips/{clip_id}.mp4"
    out_caption = f"data/raw_clips/{clip_id}.txt"

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
        r = requests.get(url, headers=headers, timeout=25, stream=True)

        if r.status_code == 200:
            with open(out_video, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16384):
                    f.write(chunk)

            size = os.path.getsize(out_video)
            if size < 10000:
                os.remove(out_video)
                return clip_id, False, "too small"

            with open(out_caption, 'w') as f:
                f.write(caption)

            return clip_id, True, size

        else:
            return clip_id, False, f"HTTP {r.status_code}"

    except Exception as e:
        if os.path.exists(out_video):
            os.remove(out_video)
        return clip_id, False, str(e)

# Run with 8 parallel workers
print(f"\nStarting parallel download with 8 workers...")

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_one, row): row for row in todo}

    with tqdm(total=len(todo), desc="Downloading") as pbar:
        for future in as_completed(futures):
            clip_id, ok, info = future.result()

            with lock:
                if ok:
                    succeeded.append(clip_id)
                else:
                    failed.append((clip_id, info))

            pbar.update(1)
            pbar.set_postfix({
                'ok': len(succeeded),
                'fail': len(failed)
            })

print(f"\nSucceeded: {len(succeeded)}")
print(f"Failed:    {len(failed)}")

# Save results
with open("download_results.json", "w") as f:
    json.dump({
        "succeeded": succeeded,
        "failed": [{"id": i, "reason": r} for i, r in failed]
    }, f, indent=2)

print("Saved download_results.json")

