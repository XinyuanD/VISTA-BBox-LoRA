import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fvd_utils import load_fvd_model, get_fvd_logits, frechet_distance


def group_frames(image_dir, num_frames=25):
    """
    Group VISTA output frames by sample ID.

    Expected filenames:
        NUSCENES_000032_0000.png
        NUSCENES_000032_0001.png
        ...
        NUSCENES_000032_0024.png
    """
    image_dir = Path(image_dir)
    groups = defaultdict(list)

    for path in sorted(image_dir.glob("*.png")):
        parts = path.stem.rsplit("_", 2)

        if len(parts) != 3:
            continue

        dataset_name, sample_id, frame_id = parts

        try:
            sample_id = int(sample_id)
            frame_id = int(frame_id)
        except ValueError:
            continue

        groups[sample_id].append((frame_id, path))

    # Sort frames inside each video
    valid_groups = {}

    for sample_id, frames in groups.items():
        frames = sorted(frames, key=lambda x: x[0])

        if len(frames) != num_frames:
            print(
                f"Warning: sample {sample_id} has "
                f"{len(frames)} frames, expected {num_frames}. Skipping."
            )
            continue

        # Make sure frame IDs are 0 ... 24
        frame_ids = [frame_id for frame_id, _ in frames]

        if frame_ids != list(range(num_frames)):
            print(
                f"Warning: sample {sample_id} has unexpected "
                f"frame IDs: {frame_ids}. Skipping."
            )
            continue

        valid_groups[sample_id] = [path for _, path in frames]

    return valid_groups


def load_video(frame_paths):
    """
    Load one 25-frame video as uint8:

        [T, H, W, C]

    The resizing to 224x224 is intentionally NOT done here.
    fvd_utils.get_fvd_logits() performs the exact LVDM
    preprocessing.
    """
    frames = []

    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.asarray(img, dtype=np.uint8))

    return np.stack(frames, axis=0)


def extract_embeddings(
    groups,
    sample_ids,
    i3d,
    device,
    batch_size=2,
    description="Extracting embeddings",
):
    """
    Process videos in small batches so we don't load the
    entire 300-video dataset into RAM/GPU memory at once.
    """
    embeddings = []

    for start in tqdm(
        range(0, len(sample_ids), batch_size),
        desc=description,
    ):
        batch_ids = sample_ids[start:start + batch_size]

        videos = [
            load_video(groups[sample_id])
            for sample_id in batch_ids
        ]

        # [B, T, H, W, C]
        videos = np.stack(videos, axis=0)

        batch_embeddings = get_fvd_logits(
            videos,
            i3d=i3d,
            device=device,
            batch_size=batch_size,
        )

        embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real_dir",
        required=True,
        help="Path to VISTA real/images directory",
    )

    parser.add_argument(
        "--generated_dir",
        required=True,
        help="Path to VISTA virtual/images directory",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of videos processed by I3D at once",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device, e.g. cuda:0 or cpu",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("Grouping real frames...")
    real_groups = group_frames(
        args.real_dir,
        num_frames=args.num_frames,
    )

    print("Grouping generated frames...")
    generated_groups = group_frames(
        args.generated_dir,
        num_frames=args.num_frames,
    )

    print(f"Complete real videos:      {len(real_groups)}")
    print(f"Complete generated videos: {len(generated_groups)}")

    # Only evaluate samples that exist completely in BOTH sets.
    sample_ids = sorted(
        set(real_groups.keys()) &
        set(generated_groups.keys())
    )

    if len(sample_ids) < 2:
        raise RuntimeError(
            "Need at least 2 complete matching videos to compute FVD."
        )

    print(f"Matching videos used:      {len(sample_ids)}")
    print(f"First sample ID:           {sample_ids[0]}")
    print(f"Last sample ID:            {sample_ids[-1]}")

    print("\nLoading I3D model...")
    i3d = load_fvd_model(device)

    print("\nExtracting real video embeddings...")
    real_embeddings = extract_embeddings(
        real_groups,
        sample_ids,
        i3d,
        device,
        batch_size=args.batch_size,
        description="Real",
    )

    print("\nExtracting generated video embeddings...")
    generated_embeddings = extract_embeddings(
        generated_groups,
        sample_ids,
        i3d,
        device,
        batch_size=args.batch_size,
        description="Generated",
    )

    print("\nEmbedding shapes:")
    print("Real:     ", real_embeddings.shape)
    print("Generated:", generated_embeddings.shape)

    print("\nComputing FVD...")

    fvd = frechet_distance(
        generated_embeddings,
        real_embeddings,
    )

    print("\n============================")
    print(f"Number of videos: {len(sample_ids)}")
    print(f"Frames per video: {args.num_frames}")
    print(f"FVD: {fvd.item():.4f}")
    print("============================")


if __name__ == "__main__":
    main()