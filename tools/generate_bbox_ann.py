import argparse
import json
import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


# ============================================================
# nuScenes category -> simplified class
# ============================================================

CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASS_NAMES)
}


def map_nuscenes_category(category_name):
    """
    Convert fine-grained nuScenes category names into the
    standard 10 detection classes.

    Returns None for categories we don't want to condition on.
    """

    if category_name.startswith("vehicle.car"):
        return "car"

    if category_name.startswith("vehicle.truck"):
        return "truck"

    if category_name.startswith("vehicle.bus"):
        return "bus"

    if category_name.startswith("vehicle.trailer"):
        return "trailer"

    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"

    if category_name.startswith("human.pedestrian"):
        return "pedestrian"

    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"

    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"

    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"

    if category_name.startswith("movable_object.barrier"):
        return "barrier"

    return None


# ============================================================
# VISTA crop geometry
# ============================================================

def get_vista_crop(
    ori_w,
    ori_h,
    target_width=1024,
    target_height=576,
):
    """
    Reproduces the geometric crop used in VISTA's load_img().

    Returns:
        crop_left
        crop_top
        crop_right
        crop_bottom
    """

    crop_left = 0
    crop_top = 0
    crop_right = ori_w
    crop_bottom = ori_h

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)

        crop_left = (ori_w - tmp_w) // 2
        crop_right = (ori_w + tmp_w) // 2

    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)

        crop_top = (ori_h - tmp_h) // 2
        crop_bottom = (ori_h + tmp_h) // 2

    return crop_left, crop_top, crop_right, crop_bottom


def transform_bbox_to_vista(
    xmin,
    ymin,
    xmax,
    ymax,
    ori_w,
    ori_h,
    target_width=1024,
    target_height=576,
):
    """
    Apply the same crop/resize geometry as VISTA and return
    normalized [xc, yc, w, h].

    Returns None if the box does not overlap the final image.
    """

    left, top, right, bottom = get_vista_crop(
        ori_w,
        ori_h,
        target_width,
        target_height,
    )

    crop_w = right - left
    crop_h = bottom - top

    # Transform from original image coordinates
    # into crop-relative coordinates.
    xmin -= left
    xmax -= left
    ymin -= top
    ymax -= top

    # Clip to crop.
    xmin = max(0.0, min(float(crop_w), xmin))
    xmax = max(0.0, min(float(crop_w), xmax))
    ymin = max(0.0, min(float(crop_h), ymin))
    ymax = max(0.0, min(float(crop_h), ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    # Resize into VISTA target resolution.
    scale_x = target_width / crop_w
    scale_y = target_height / crop_h

    xmin *= scale_x
    xmax *= scale_x
    ymin *= scale_y
    ymax *= scale_y

    # Convert xyxy -> normalized cxcywh.
    xc = ((xmin + xmax) / 2.0) / target_width
    yc = ((ymin + ymax) / 2.0) / target_height

    w = (xmax - xmin) / target_width
    h = (ymax - ymin) / target_height

    return [
        float(xc),
        float(yc),
        float(w),
        float(h),
    ]


# ============================================================
# Projection
# ============================================================

def project_box_to_2d(box, camera_intrinsic):
    """
    Project a camera-coordinate nuScenes 3D Box into a 2D
    enclosing rectangle.

    Returns:
        xmin, ymin, xmax, ymax

    or None if the box is completely behind the camera.
    """

    corners = box.corners()  # shape (3, 8)

    # z is depth in the camera coordinate frame.
    in_front = corners[2, :] > 1e-5

    if not np.any(in_front):
        return None

    corners = corners[:, in_front]

    projected = view_points(
        corners,
        np.asarray(camera_intrinsic),
        normalize=True,
    )

    xs = projected[0, :]
    ys = projected[1, :]

    xmin = float(xs.min())
    ymin = float(ys.min())
    xmax = float(xs.max())
    ymax = float(ys.max())

    return xmin, ymin, xmax, ymax


# ============================================================
# Main per-frame preprocessing
# ============================================================

def get_bbox_annotation(
    nusc,
    sample_data_record,
    target_width,
    target_height,
    allowed_visibility={"3", "4"},
):
    """
    Generate bbox tokens for one VISTA conditioning frame.

    Output format:
        [
            [class_id, xc, yc, w, h],
            ...
        ]
    """

    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        sample_data_record["token"]
    )

    with Image.open(image_path) as image:
        ori_w, ori_h = image.size

    output = []

    for box in boxes:

        # --------------------------------------------------
        # Visibility filter
        # --------------------------------------------------

        ann = nusc.get(
            "sample_annotation",
            box.token,
        )

        visibility = ann["visibility_token"]

        if visibility not in allowed_visibility:
            continue

        # --------------------------------------------------
        # Category mapping
        # --------------------------------------------------

        class_name = map_nuscenes_category(box.name)

        if class_name is None:
            continue

        class_id = CLASS_TO_ID[class_name]

        # --------------------------------------------------
        # Project 3D -> 2D
        # --------------------------------------------------

        projected = project_box_to_2d(
            box,
            camera_intrinsic,
        )

        if projected is None:
            continue

        xmin, ymin, xmax, ymax = projected

        # --------------------------------------------------
        # Apply VISTA crop/resize and normalize
        # --------------------------------------------------

        bbox = transform_bbox_to_vista(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            ori_w=ori_w,
            ori_h=ori_h,
            target_width=target_width,
            target_height=target_height,
        )

        if bbox is None:
            continue

        xc, yc, w, h = bbox

        output.append([
            class_id,
            xc,
            yc,
            w,
            h,
        ])

    return output


# ============================================================
# Main
# ============================================================

def main(args):

    print("Loading nuScenes...")

    nusc = NuScenes(
        version=args.version,
        dataroot=args.nuscenes_root,
        verbose=True,
    )

    print(f"Loading VISTA annotations: {args.input_json}")

    with open(args.input_json, "r") as f:
        vista_data = json.load(f)

    print(f"Number of VISTA clips: {len(vista_data)}")

    # ------------------------------------------------------
    # Build filename -> sample_data lookup ONCE.
    #
    # Avoid searching 2.6 million sample_data records
    # separately for every VISTA clip.
    # ------------------------------------------------------

    print("Building sample_data filename lookup...")

    sample_data_by_filename = {}

    for sd in tqdm(nusc.sample_data):
        sample_data_by_filename[sd["filename"]] = sd

    print(
        f"Indexed {len(sample_data_by_filename)} sample_data records."
    )

    # ------------------------------------------------------
    # Process VISTA entries
    # ------------------------------------------------------

    processed = []
    missing_frames = []
    empty_bbox_count = 0

    box_count = 0
    class_counter = Counter()

    for item in tqdm(vista_data, desc="Generating bounding boxes"):

        if len(item["frames"]) == 0:
            raise ValueError("Encountered clip with no frames.")

        # VISTA conditioning frame
        cond_frame = item["frames"][0]

        sd = sample_data_by_filename.get(cond_frame)

        if sd is None:
            missing_frames.append(cond_frame)

            new_item = dict(item)
            new_item["bboxes"] = []

            processed.append(new_item)
            continue

        # Sanity check: bbox GT should come from keyframes.
        if not sd["is_key_frame"]:
            print(
                f"\nWARNING: conditioning frame is not a keyframe:\n"
                f"{cond_frame}"
            )

        bboxes = get_bbox_annotation(
            nusc=nusc,
            sample_data_record=sd,
            target_width=args.target_width,
            target_height=args.target_height,
        )

        if len(bboxes) == 0:
            empty_bbox_count += 1

        for bbox in bboxes:
            class_id = bbox[0]

            class_counter[
                CLASS_NAMES[class_id]
            ] += 1

        box_count += len(bboxes)

        new_item = dict(item)
        new_item["bboxes"] = bboxes

        processed.append(new_item)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output_json) or ".",
        exist_ok=True,
    )

    with open(args.output_json, "w") as f:
        json.dump(
            processed,
            f,
            indent=2,
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Clips:                  {len(processed)}")
    print(f"Total boxes:            {box_count}")
    print(
        f"Average boxes / clip:   "
        f"{box_count / max(len(processed), 1):.2f}"
    )

    print(
        f"Clips with zero boxes:  "
        f"{empty_bbox_count}"
    )

    print(
        f"Missing conditioning frames: "
        f"{len(missing_frames)}"
    )

    print("\nClass distribution:")

    for class_name in CLASS_NAMES:
        print(
            f"  {class_name:22s}: "
            f"{class_counter[class_name]}"
        )

    print(f"\nOutput written to:")
    print(args.output_json)

    if missing_frames:
        missing_file = args.output_json + ".missing.txt"

        with open(missing_file, "w") as f:
            for filename in missing_frames:
                f.write(filename + "\n")

        print(
            f"\nMissing filenames saved to:"
            f"\n{missing_file}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nuscenes-root",
        required=True,
        help="Path to nuScenes dataset root",
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Original VISTA nuScenes annotation JSON",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="New JSON containing bbox annotations",
    )

    parser.add_argument(
        "--version",
        default="v1.0-trainval",
    )

    parser.add_argument(
        "--target-height",
        type=int,
        default=576,
    )

    parser.add_argument(
        "--target-width",
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    main(args)