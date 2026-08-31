import os

import torch

from .common import BaseDataset


def balance_with_actions(samples, increase_factor=5, exceptions=None):
    if exceptions is None:
        exceptions = [2, 3]
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if each_sample["cmd"] not in exceptions:
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


def resample_complete_samples(samples, increase_factor=5):
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if (each_sample["speed"] and each_sample["angle"] and each_sample["z"] > 0
                    and 0 < each_sample["goal"][0] < 1600 and 0 < each_sample["goal"][1] < 900):
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


class NuScenesDataset(BaseDataset):
    def __init__(self, data_root="vwm/data/nuscenes", 
                 anno_file="vwm/data/annos/nuScenes_bbox.json",
                 target_height=320, target_width=576, num_frames=25, max_objects=32):
        if not os.path.exists(data_root):
            raise ValueError("Cannot find dataset {}".format(data_root))
        if not os.path.exists(anno_file):
            raise ValueError("Cannot find annotation {}".format(anno_file))
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        print("nuScenes loaded:", len(self))
        self.samples = balance_with_actions(self.samples, increase_factor=5)
        print("nuScenes balanced:", len(self))
        self.samples = resample_complete_samples(self.samples, increase_factor=2)
        print("nuScenes resampled:", len(self))
        self.action_mod = 0
        
        # bbox
        self.max_objects = max_objects

    # bbox
    def get_bbox_condition(self, sample_dict):
        bboxes = sample_dict.get("bboxes", [])

        # Keep largest boxes if there are too many.
        if len(bboxes) > self.max_objects:
            bboxes = sorted(
                bboxes,
                key=lambda b: b[3] * b[4],
                reverse=True,
            )[:self.max_objects]

        bbox_classes = torch.zeros(
            self.max_objects,
            dtype=torch.long,
        )

        bbox_coords = torch.zeros(
            (self.max_objects, 4),
            dtype=torch.float32,
        )

        bbox_valid_mask = torch.zeros(
            self.max_objects,
            dtype=torch.bool,
        )

        if len(bboxes) > 0:
            bboxes = torch.tensor(
                bboxes,
                dtype=torch.float32,
            )

            n = bboxes.shape[0]

            # first column = class id
            bbox_classes[:n] = bboxes[:, 0].long()

            # remaining columns = normalized xc, yc, w, h
            bbox_coords[:n] = bboxes[:, 1:5]

            bbox_valid_mask[:n] = True

        return bbox_classes, bbox_coords, bbox_valid_mask
    
    # bbox
    def adjust_bbox_for_current_resolution(self, bbox_coords, bbox_valid_mask):
        # Current annotations were generated for 1024x576 / 16:9.
        # No adjustment needed for 1024x576.
        if self.target_width == 1024 and self.target_height == 576:
            return bbox_coords

        # Specific correction for VISTA stage-1 576x320.
        if self.target_width == 576 and self.target_height == 320:
            coords = bbox_coords.clone()

            valid = bbox_valid_mask
            boxes = coords[valid]

            if boxes.numel() == 0:
                return coords

            # normalized cx, cy, w, h in original 1600x900 frame
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            bw = boxes[:, 2]
            bh = boxes[:, 3]

            # Convert vertical extent back to original-image pixels.
            y1 = (cy - bh / 2.0) * 900.0
            y2 = (cy + bh / 2.0) * 900.0

            # Stage-1 VISTA crop is y=[6, 894).
            y1 = torch.clamp(y1, 6.0, 894.0)
            y2 = torch.clamp(y2, 6.0, 894.0)

            # Convert to coordinates relative to cropped 888px image.
            y1 = (y1 - 6.0) / 888.0
            y2 = (y2 - 6.0) / 888.0

            boxes[:, 1] = (y1 + y2) / 2.0
            boxes[:, 3] = y2 - y1

            coords[valid] = boxes
            return coords

        raise ValueError(
            f"Unsupported bbox target resolution: "
            f"{self.target_width}x{self.target_height}"
        )
    
    def get_image_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["frames"][current_index])

    def build_data_dict(self, image_seq, sample_dict):
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        
        # bbox
        bbox_classes, bbox_coords, bbox_valid_mask = self.get_bbox_condition(sample_dict)
        bbox_coords = self.adjust_bbox_for_current_resolution(bbox_coords, bbox_valid_mask)

        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug,
            
            # bbox
            "bbox_classes": bbox_classes,
            "bbox_coords": bbox_coords,
            "bbox_valid_mask": bbox_valid_mask
        }
        if self.action_mod == 0:
            data_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
        elif self.action_mod == 1:
            data_dict["command"] = torch.tensor(sample_dict["cmd"])
        elif self.action_mod == 2:
            # scene might be empty
            if sample_dict["speed"]:
                data_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
            # scene might be empty
            if sample_dict["angle"]:
                data_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
        elif self.action_mod == 3:
            # point might be invalid
            if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                data_dict["goal"] = torch.tensor([
                    sample_dict["goal"][0] / 1600,
                    sample_dict["goal"][1] / 900
                ])
        else:
            raise ValueError
        return data_dict

    def __getitem__(self, index):
        sample_dict = self.samples[index]
        self.action_mod = (self.action_mod + index) % 4

        image_seq = list()
        for i in range(self.num_frames):
            current_index = i
            img_path = self.get_image_path(sample_dict, current_index)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
        return self.build_data_dict(image_seq, sample_dict)
