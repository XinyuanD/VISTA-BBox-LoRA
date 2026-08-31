import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to full Lightning training checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output path for BBox-only safetensors checkpoint",
    )

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    if "state_dict" not in checkpoint:
        raise RuntimeError(
            "Expected a Lightning checkpoint containing 'state_dict'."
        )

    state_dict = checkpoint["state_dict"]

    bbox_state_dict = {
        key: value.contiguous()
        for key, value in state_dict.items()
        if "bbox" in key
    }

    if len(bbox_state_dict) == 0:
        raise RuntimeError(
            "No parameters containing 'bbox' were found."
        )

    total_params = sum(
        tensor.numel()
        for tensor in bbox_state_dict.values()
    )

    print(f"BBox tensors: {len(bbox_state_dict)}")
    print(f"BBox parameters: {total_params:,}")

    print("\nExporting:")

    for key in sorted(bbox_state_dict):
        tensor = bbox_state_dict[key]
        print(
            f"  {key}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}"
        )

    save_file(
        bbox_state_dict,
        args.output,
        metadata={
            "base_model": "OpenDriveLab/VISTA",
            "description": "VISTA BBox LoRA 5K weights",
        },
    )

    print(f"\nSaved BBox weights to: {args.output}")


if __name__ == "__main__":
    main()