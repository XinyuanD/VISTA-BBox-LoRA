# VISTA BBox LoRA

Bounding box LoRA for improving object-level temporal consistency in [VISTA](https://github.com/OpenDriveLab/Vista).

This repository is based on the official OpenDriveLab VISTA implementation and extends it with explicit bounding-box conditioning for road agents. Bounding boxes from the conditioning frame are encoded into spatially localized object features and injected into VISTA's temporal cross-attention through a dedicated LoRA pathway.

The goal is to improve the temporal persistence of vehicles, pedestrians, and other road agents without large-scale retraining of the pretrained VISTA model.

## Results

The final model was selected using a fixed 300-sample nuScenes checkpoint-selection set and then evaluated on a separate held-out set of 300 previously unseen samples.

### Held-Out Test Set

| Model | FID ↓ | FVD ↓ | Agent Consistency ↑ |
| --- | ---: | ---: | ---: |
| VISTA | 23.23 | 335.93 | 0.70 |
| **BBox LoRA 5K** | **20.75** | **268.07** | **0.71** |

On the held-out test set, BBox LoRA improves FID by approximately **10.7%** and FVD by approximately **20.2%**, while also improving Agent Consistency.

### Checkpoint Selection Set

| Model | FID ↓ | FVD ↓ | Agent Consistency ↑ |
| --- | ---: | ---: | ---: |
| Real nuScenes | — | — | 0.83 |
| VISTA | 16.41 | 306.74 | 0.69 |
| BBox LoRA 3K | 20.40 | 269.53 | 0.60 |
| **BBox LoRA 5K** | 20.06 | **259.66** | 0.71 |
| BBox LoRA 7K | 21.77 | 271.53 | 0.70 |
| BBox LoRA 10K | 20.30 | 267.16 | 0.70 |
| BBox LoRA 15K | 20.62 | 287.64 | **0.72** |

The 5K checkpoint provides the best overall tradeoff across the evaluated metrics. Longer training does not produce further improvements in overall video quality.

## Method

For a detailed description of the bounding-box encoder, dense spatial representation, temporal cross-attention modification, LoRA design, evaluation setup, and experiment analysis, see the full [experiment report](docs/VISTA_BBox_LoRA_Experiment_Report.pdf).

## Setup

This repository uses the same environment and dependencies as the official [VISTA implementation](https://github.com/OpenDriveLab/Vista). Please follow the installation instructions in the original repository.

The original pretrained VISTA checkpoint and nuScenes dataset are also required. The copied upstream documentation in [`docs/upstream/`](docs/upstream/) is retained for reference only.

## Bounding Box Annotation Generation

This repository uses bounding-box annotations derived from the original VISTA nuScenes annotations. For convenience, the pre-generated validation annotation file is included:

`vwm/data/annos/nuScenes_bbox_val.json`

The full training annotation file can be generated locally from the original VISTA nuScenes annotations using:

```bash
python tools/generate_bbox_ann.py \
  --nuscenes-root <path-to-nuscenes> \
  --input-json <path-to-original-vista-annotation-json> \
  --output-json <path-to-output-bbox-json>
```

## Training

The BBox LoRA model is trained using the Stage-1 VISTA configuration at 320 × 576 resolution. The final experiment used the training configuration in `configs/training/vista_phase2_stage1.yaml`.

Run training with:

```bash
python train.py \
  --base configs/training/vista_phase2_stage1.yaml \
  --finetune ckpts/vista.safetensors \
  --num_nodes 1 \
  --n_devices 1 \
  --logdir logs/bbox_lora_stage1_5k
```

After training, the BBox-specific parameters can be extracted from the full checkpoint into a lightweight safetensors file:

```bash
python tools/export_bbox_weights.py \
  --checkpoint <path-to-your-5k.ckpt> \
  --output ckpts/bbox_lora_5k.safetensors
```

## Inference

The 5K-step BBox LoRA checkpoint used for the reported experiments is included at `ckpts/bbox_lora_5k.safetensors`. It contains only the BBox-specific trainable parameters (~8.2M parameters) and is loaded on top of the original pretrained VISTA weights.

Place the checkpoints at:

```text
ckpts/
├── vista.safetensors
└── bbox_lora_5k.safetensors
```

For reproducible evaluation, use the provided index files:

- `eval_indices_300.txt` for the checkpoint-selection set
- `test_indices_300.txt` for the held-out test set

Run BBox LoRA inference with:

```bash
python sample.py \
  --bbox_ckpt ckpts/bbox_lora_5k.safetensors \
  --height 320 \
  --width 576 \
  --save outputs/nuscenes_bbox_lora_5k \
  --indices_file test_indices_300.txt
```

For the baseline comparison, run the original VISTA checkpoint without ```--bbox_ckpt``` while keeping the same sampling settings and indices. The held-out comparison uses native 320 × 576 generation for both VISTA and BBox LoRA.

## Evaluation

The experiments report FID, FVD, and Agent Consistency.

### FID

FID is computed using `pytorch-fid`:

```bash
python -m pytorch_fid \
    /path/to/real_frames \
    /path/to/generated_frames \
    --batch-size 64
```

### FVD

FVD is computed using the evaluation code under `benchmark/`:

```bash
python benchmark/compute_fvd.py \
    --real_dir /path/to/real_videos \
    --generated_dir /path/to/generated_videos \
    --i3d_path /path/to/i3d_pretrained_400.pt \
    --batch_size 2
```

### Agent Consistency

Agent Consistency is evaluated using the metric implementation from [DrivingGen](https://github.com/NVlabs/DrivingGen).

Because the evaluator depends on the DrivingGen codebase and environment, this metric is run separately from the VISTA environment. We use a custom `benchmark/agent_consistency_wrapper.py` wrapper placed inside a local DrivingGen checkout to prepare VISTA-generated sequences and invoke the DrivingGen evaluator.

Typical workflow:

1. Generate samples using this repository in the VISTA environment.
2. Switch to the DrivingGen conda environment.
3. Run the custom Agent Consistency wrapper from within the DrivingGen repository.

```bash
conda activate <drivinggen-env>
cd <path-to-DrivingGen>

python agent_consistency_wrapper.py \
  --input /path/to/generated/frames \
  --real_frame0_dir /path/to/real/frames \
  --yolo_checkpoint ckpt/yolov10x.pt \
  --samurai_checkpoint third_parties/samurai/sam2/checkpoints/sam2.1_hiera_large.pt \
  --conf 0.3 \
  --work_dir /path/to/work_dir \
  --output /path/to/results.csv
```
## Acknowledgements

This repository is based on the official [OpenDriveLab VISTA](https://github.com/OpenDriveLab/Vista) implementation. VISTA and the original source code were developed by OpenDriveLab. This repository contains modifications for bounding-box conditioning and BBox LoRA experiments.

Agent Consistency evaluation is based on the [DrivingGen](https://github.com/youngzhou1999/DrivingGen) evaluation framework.

## License

This project is based on VISTA and retains the original Apache License 2.0. See [`LICENSE`](LICENSE) for details.