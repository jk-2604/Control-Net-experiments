from datasets import load_dataset
import csv
import json
from tqdm import tqdm

print("Streaming WebVid metadata — collecting 5000 clips...")

ds = load_dataset(
    "TempoFunk/webvid-10M",
    split="train",
    streaming=True
)

rows = []
skipped = 0

for row in tqdm(ds, desc="Scanning"):
    duration = row.get('duration', 'PT00H00M00S')

    # Filter: keep clips longer than 4 seconds
    # Duration string format PT00H00M13S — compare lexicographically
    # PT00H00M04S is 4 seconds — anything greater passes
    if duration <= 'PT00H00M04S':
        skipped += 1
        continue

    # Filter: must have a real caption
    caption = row.get('name', '').strip()
    if len(caption) < 10:
        skipped += 1
        continue

    rows.append({
        'videoid':    row['videoid'],
        'contentUrl': row['contentUrl'],
        'duration':   duration,
        'caption':    caption
    })

    if len(rows) % 500 == 0:
        print(f"  Collected {len(rows)} / 5000  (skipped {skipped})")

    if len(rows) >= 5000:
        break

print(f"\nFinal: {len(rows)} rows collected, {skipped} skipped")

# Save
with open("webvid_5k.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['videoid','contentUrl','duration','caption'])
    writer.writeheader()
    writer.writerows(rows)

with open("webvid_5k_summary.json", "w") as f:
    json.dump({"total": len(rows), "skipped": skipped}, f, indent=2)

print("Saved webvid_5k.csv")
