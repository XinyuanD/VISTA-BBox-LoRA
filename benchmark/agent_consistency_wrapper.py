# MUST be run from a DrivingGen checkout and environment

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# DrivingGen third parties
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_parties"))

from ultralytics import YOLOv10
from samurai.scripts.demo import samurai_main

# DrivingGen metric
sys.path.insert(0, str(ROOT / "drivinggen"))
from videos.video_a_consist import stability_metric


ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def collect_frame_clips(input_dir):
    """
    Groups flat files like:
      NUSCENES_000032_0000.png
      NUSCENES_000032_0001.png
      ...
      NUSCENES_000033_0000.png

    Returns:
      {
        "NUSCENES_000032": [frame paths...],
        "NUSCENES_000033": [frame paths...],
      }
    """
    clips = {}

    for p in sorted(Path(input_dir).glob("*.png")):
        stem = p.stem

        # split off final _FRAMEID
        scene_name, frame_id = stem.rsplit("_", 1)

        clips.setdefault(scene_name, []).append(
            (int(frame_id), str(p))
        )

    # sort frames numerically inside each scene
    for scene_name in clips:
        clips[scene_name] = [
            path
            for _, path in sorted(clips[scene_name])
        ]

    return clips


def get_real_frame0_path(scene_name, real_frame0_dir):
    if real_frame0_dir is None:
        return None

    real_frame0_dir = Path(real_frame0_dir)

    candidates = [
        real_frame0_dir / f"{scene_name}_0000.png",
        real_frame0_dir / f"{scene_name}_0000.jpg",
        real_frame0_dir / f"{scene_name}_0000.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not find real frame 0 for {scene_name} in "
        f"{real_frame0_dir}"
    )


def evaluate_frame_clip(
    scene_name,
    frame_paths,
    detector,
    samurai_checkpoint,
    work_dir,
    conf_threshold,
    real_frame0_dir=None,
):
    if len(frame_paths) < 2:
        return [], []

    metric_img_dir = make_metric_frame_dir(
        scene_name,
        frame_paths,
        work_dir,
    )
    
    real_frame0_path = get_real_frame0_path(
        scene_name,
        real_frame0_dir,
    )
    detection_frame = (
        real_frame0_path
        if real_frame0_path is not None
        else frame_paths[0]
    )

    initial_agents = detect_initial_agents(
        detector,
        detection_frame,
        conf_threshold=conf_threshold,
    )

    print(
        f"{scene_name}: YOLO detected "
        f"{len(initial_agents)} initial road agents"
    )

    agent_rows = []
    scores = []

    for agent_idx, agent in enumerate(initial_agents):

        embed_dir = (
            work_dir
            / "embeddings"
            / scene_name
            / f"agent_{agent_idx}.pkl"
        )

        embed_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            result = evaluate_agent(
                frame_paths=frame_paths,
                initial_bbox=agent["bbox"],
                label=agent["class_name"],
                samurai_checkpoint=samurai_checkpoint,
                embed_dir=embed_dir,
                metric_img_dir=metric_img_dir
            )

        except Exception as e:
            print(
                f"Warning: failed tracking "
                f"{scene_name} agent {agent_idx}: {e}"
            )
            continue

        if result is None:
            continue

        row = {
            "video": scene_name,
            "agent_id": agent_idx,
            "class_id": agent["class_id"],
            "class_name": agent["class_name"],
            "detection_confidence": agent["confidence"],
            "initial_x1": agent["bbox"][0],
            "initial_y1": agent["bbox"][1],
            "initial_x2": agent["bbox"][2],
            "initial_y2": agent["bbox"][3],
            "num_tracked_frames": result["num_tracked_frames"],
            "R": result["R"],
            "A": result["A"],
            "agent_consistency": result["score"],
        }

        agent_rows.append(row)
        scores.append(result["score"])

    return agent_rows, scores

def make_metric_frame_dir(scene_name, frame_paths, work_dir):
    """
    Create a per-scene directory with DrivingGen-style filenames:

        00000.png
        00001.png
        ...

    Uses symlinks, so no image data is duplicated.
    """
    metric_dir = work_dir / "metric_frames" / scene_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(frame_paths):
        src = Path(src)

        dst = metric_dir / f"{i:05d}{src.suffix.lower()}"

        if not dst.exists():
            dst.symlink_to(src.resolve())

    return metric_dir

def get_frame_paths(video_path, temp_dir):
    """
    Extract a video into numbered JPG frames because SAMURAI works
    naturally with frame paths.
    """

    temp_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        path = temp_dir / f"{frame_idx:05d}.jpg"

        cv2.imwrite(str(path), frame)

        frame_paths.append(str(path))
        frame_idx += 1

    cap.release()

    return frame_paths


def detect_initial_agents(
    model,
    frame_path,
    conf_threshold=0.3,
):
    """
    Detect road agents in frame 0 using YOLOv10.

    Returns:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": ...,
                "class_name": ...,
                "confidence": ...
            },
            ...
        ]
    """

    result = model(frame_path, verbose=False)[0]

    agents = []

    if result.boxes is None:
        return agents

    for box in result.boxes:

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if class_id not in ROAD_CLASSES:
            continue

        if confidence < conf_threshold:
            continue

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = [
            int(v)
            for v in xyxy
        ]

        agents.append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": ROAD_CLASSES[class_id],
                "confidence": confidence,
            }
        )

    return agents


def evaluate_agent(
    frame_paths,
    initial_bbox,
    label,
    samurai_checkpoint,
    embed_dir,
    metric_img_dir=None
):
    """
    Track one object with SAMURAI, then compute DrivingGen
    Agent Consistency.
    """

    _, tracked_boxes = samurai_main(
        model_path=samurai_checkpoint,
        video_path=frame_paths,
        txt_path=initial_bbox,
        save_to_video=False,
    )

    if len(tracked_boxes) < 2:
        return None

    if metric_img_dir is None:
        metric_img_dir = Path(frame_paths[0]).parent

    result = stability_metric(
        img_dir=str(metric_img_dir),
        boxes=tracked_boxes,
        label=label,
        embed_dir=str(embed_dir),
    )

    return {
        "R": result["R"],
        "A": result["A"],
        "score": result["score"],
        "num_tracked_frames": len(tracked_boxes),
    }


def evaluate_video(
    video_path,
    detector,
    samurai_checkpoint,
    work_dir,
    conf_threshold,
    real_frame0_dir=None,
):
    video_name = video_path.stem

    frame_dir = (
        work_dir
        / "frames"
        / video_name
    )

    frame_paths = get_frame_paths(
        video_path,
        frame_dir,
    )

    if len(frame_paths) < 2:
        return [], []

    real_frame0_path = get_real_frame0_path(
        video_name,
        real_frame0_dir,
    )
    detection_frame = (
        real_frame0_path
        if real_frame0_path is not None
        else frame_paths[0]
    )

    initial_agents = detect_initial_agents(
        detector,
        detection_frame,
        conf_threshold=conf_threshold,
    )
    
    print(
        f"{video_name}: YOLO detected "
        f"{len(initial_agents)} initial road agents"
    )

    for agent in initial_agents:
        print(
            agent["class_name"],
            f"conf={agent['confidence']:.3f}",
            agent["bbox"],
        )

    agent_rows = []
    scores = []

    for agent_idx, agent in enumerate(initial_agents):

        embed_dir = (
            work_dir
            / "embeddings"
            / video_name
            / f"agent_{agent_idx}.pkl"
        )

        embed_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            result = evaluate_agent(
                frame_paths=frame_paths,
                initial_bbox=agent["bbox"],
                label=agent["class_name"],
                samurai_checkpoint=samurai_checkpoint,
                embed_dir=embed_dir,
            )

        except Exception as e:
            print(
                f"Warning: failed tracking "
                f"{video_name} agent {agent_idx}: {e}"
            )
            continue

        if result is None:
            continue

        row = {
            "video": video_path.name,
            "agent_id": agent_idx,
            "class_id": agent["class_id"],
            "class_name": agent["class_name"],
            "detection_confidence": agent["confidence"],
            "initial_x1": agent["bbox"][0],
            "initial_y1": agent["bbox"][1],
            "initial_x2": agent["bbox"][2],
            "initial_y2": agent["bbox"][3],
            "num_tracked_frames": result[
                "num_tracked_frames"
            ],
            "R": result["R"],
            "A": result["A"],
            "agent_consistency": result["score"],
        }

        agent_rows.append(row)
        scores.append(result["score"])

    return agent_rows, scores


def main(args):

    input_dir = Path(args.input)
    work_dir = Path(args.work_dir)

    videos = sorted(
        list(input_dir.glob("*.mp4"))
        + list(input_dir.glob("*.avi"))
        + list(input_dir.glob("*.mov"))
    )
    
    frame_clips = collect_frame_clips(input_dir)

    if len(videos) == 0 and len(frame_clips) == 0:
        raise RuntimeError(
            f"No videos or frame clips found in {input_dir}"
        )

    print(
        f"Found {len(videos)} videos and "
        f"{len(frame_clips)} frame clips"
    )

    print("Loading YOLOv10...")
    detector = YOLOv10(args.yolo_checkpoint)

    all_agent_rows = []
    all_video_rows = []

    all_scores = []

    for video_path in tqdm(videos):

        agent_rows, scores = evaluate_video(
            video_path=video_path,
            detector=detector,
            samurai_checkpoint=args.samurai_checkpoint,
            work_dir=work_dir,
            conf_threshold=args.conf,
            real_frame0_dir=args.real_frame0_dir,
        )

        all_agent_rows.extend(agent_rows)
        all_scores.extend(scores)

        if len(scores) > 0:
            video_score = float(
                np.mean(scores)
            )
        else:
            video_score = float("nan")

        all_video_rows.append(
            {
                "video": video_path.name,
                "num_agents": len(scores),
                "agent_consistency": video_score,
            }
        )
    
    # Flat-frame inputs
    for scene_name, frame_paths in tqdm(frame_clips.items()):
        agent_rows, scores = evaluate_frame_clip(
            scene_name=scene_name,
            frame_paths=frame_paths,
            detector=detector,
            samurai_checkpoint=args.samurai_checkpoint,
            work_dir=work_dir,
            conf_threshold=args.conf,
            real_frame0_dir=args.real_frame0_dir,
        )

        all_agent_rows.extend(agent_rows)
        all_scores.extend(scores)

        video_score = (
            float(np.mean(scores))
            if scores
            else float("nan")
        )

        all_video_rows.append(
            {
                "video": scene_name,
                "num_agents": len(scores),
                "agent_consistency": video_score,
            }
        )

    #
    # DrivingGen averages agents within scenes,
    # then averages across scenes.
    #
    valid_video_scores = [
        row["agent_consistency"]
        for row in all_video_rows
        if not np.isnan(
            row["agent_consistency"]
        )
    ]

    final_score = (
        float(np.mean(valid_video_scores))
        if valid_video_scores
        else float("nan")
    )

    print()
    print("=" * 60)
    print("DrivingGen Agent Consistency")
    print("=" * 60)

    print(
        f"Videos evaluated:          "
        f"{len(videos)}"
    )

    print(
        f"Videos with valid agents:  "
        f"{len(valid_video_scores)}"
    )

    print(
        f"Total agents evaluated:    "
        f"{len(all_scores)}"
    )

    print(
        f"Agent Consistency:         "
        f"{final_score:.4f}"
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Video-level CSV
    #
    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "num_agents",
                "agent_consistency",
            ],
        )

        writer.writeheader()
        writer.writerows(all_video_rows)

    #
    # Agent-level CSV
    #
    agent_output = output_path.with_name(
        output_path.stem
        + "_agents.csv"
    )

    agent_fields = [
        "video",
        "agent_id",
        "class_id",
        "class_name",
        "detection_confidence",
        "initial_x1",
        "initial_y1",
        "initial_x2",
        "initial_y2",
        "num_tracked_frames",
        "R",
        "A",
        "agent_consistency",
    ]

    with open(
        agent_output,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=agent_fields,
        )

        writer.writeheader()
        writer.writerows(all_agent_rows)

    print()
    print(
        f"Video results: {output_path}"
    )
    print(
        f"Agent results: {agent_output}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing videos",
    )

    parser.add_argument(
        "--real_frame0_dir",
        default=None,
        help=(
            "Optional directory containing matching real frame-0 images "
            "named like NUSCENES_000032_0000.png. When provided, YOLO "
            "initial detections are taken from these real frames."
        ),
    )

    parser.add_argument(
        "--yolo_checkpoint",
        required=True,
        help="Path to local yolov10x.pt",
    )

    parser.add_argument(
        "--samurai_checkpoint",
        required=True,
        help="Path to SAM2.1 checkpoint",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO detection threshold",
    )

    parser.add_argument(
        "--work_dir",
        default="./cache/agent_consistency",
    )

    parser.add_argument(
        "--output",
        default="./agent_consistency.csv",
    )

    args = parser.parse_args()

    main(args)