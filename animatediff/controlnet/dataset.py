import torch
from torch.utils.data import Dataset
import os
import json


class DepthVideoDataset(Dataset):
    def __init__(self, processed_dir, split="train",
                 val_fraction=0.05, clip_list=None):
        self.processed_dir = processed_dir

        if clip_list is not None:
            all_clips = sorted([
                c for c in clip_list
                if os.path.exists(os.path.join(processed_dir, c, "video.pt"))
                and os.path.exists(os.path.join(processed_dir, c, "depth.pt"))
                and os.path.exists(os.path.join(processed_dir, c, "metadata.json"))
            ])
        else:
            all_clips = sorted([
                d for d in os.listdir(processed_dir)
                if os.path.isdir(os.path.join(processed_dir, d))
                and os.path.exists(os.path.join(processed_dir, d, "video.pt"))
                and os.path.exists(os.path.join(processed_dir, d, "depth.pt"))
                and os.path.exists(os.path.join(processed_dir, d, "metadata.json"))
            ])

        n_val = max(1, int(len(all_clips) * val_fraction))
        if split == "train":
            self.clips = all_clips[n_val:]
        else:
            self.clips = all_clips[:n_val]

        print(f"Dataset [{split}]: {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_id  = self.clips[idx]
        clip_dir = os.path.join(self.processed_dir, clip_id)

        video = torch.load(
            os.path.join(clip_dir, "video.pt"),
            map_location="cpu"
        ).float()

        depth = torch.load(
            os.path.join(clip_dir, "depth.pt"),
            map_location="cpu"
        ).float()

        with open(os.path.join(clip_dir, "metadata.json")) as f:
            meta = json.load(f)
        caption = meta.get("caption", "")

        # 4 evenly spaced frames
        indices = [0, 2, 4, 6]
        video = video[indices]
        depth = depth[indices]

        # [F, C, H, W] → [C, F, H, W]
        video = video.permute(1, 0, 2, 3)
        depth = depth.permute(1, 0, 2, 3)

        return {
            "video":   video,
            "depth":   depth,
            "caption": caption
        }