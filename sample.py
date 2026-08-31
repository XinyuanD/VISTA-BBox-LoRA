import argparse
import json
import random

from pytorch_lightning import seed_everything
from torchvision import transforms

import init_proj_path
from sample_utils import *

VERSION2SPECS = {
    "vwm": {
        "config": "configs/inference/vista.yaml",
        "ckpt": "ckpts/vista.safetensors"
    }
}

DATASET2SOURCES = {
    "NUSCENES": {
        "data_root": "vwm/data/nuscenes",
        "anno_file": "vwm/data/annos/nuScenes_bbox_val.json"
    },
    "IMG": {
        "data_root": "image_folder"
    }
}


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--version",
        type=str,
        default="vwm",
        help="model version"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NUSCENES",
        help="dataset name"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs",
        help="directory to save samples"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="free",
        help="action mode for control, such as traj, cmd, steer, goal"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=1,
        help="number of sampling rounds"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=25,
        help="number of frames for each round"
    )
    parser.add_argument(
        "--n_conds",
        type=int,
        default=1,
        help="number of initial condition frames for the first round"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="random seed for seed_everything"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="target height of the generated video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="target width of the generated video"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="scale of the classifier-free guidance"
    )
    parser.add_argument(
        "--cond_aug",
        type=float,
        default=0.0,
        help="strength of the noise augmentation"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="number of sampling steps"
    )
    parser.add_argument(
        "--rand_gen",
        action="store_false",
        help="whether to generate samples randomly or sequentially"
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="whether to save memory or not"
    )
    parser.add_argument(
        "--indices_file",
        type=str,
        default=None,
        help="optional text file containing dataset indices to generate"
    )
    
    parser.add_argument(
        "--bbox_ckpt",
        type=str,
        default=None,
        help="path to BBox LoRA safetensors checkpoint"
    )
        
    return parser

# bbox
def get_bbox_condition(sample_dict, max_objects=32):
    bboxes = sample_dict.get("bboxes", [])

    # Same rule as training: if too many objects,
    # keep the largest boxes by area.
    if len(bboxes) > max_objects:
        bboxes = sorted(
            bboxes,
            key=lambda b: b[3] * b[4],
            reverse=True,
        )[:max_objects]

    bbox_classes = torch.zeros(max_objects, dtype=torch.long)
    bbox_coords = torch.zeros((max_objects, 4), dtype=torch.float32)
    bbox_valid_mask = torch.zeros(max_objects, dtype=torch.bool)

    if len(bboxes) > 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        n = bboxes.shape[0]

        bbox_classes[:n] = bboxes[:, 0].long()
        bbox_coords[:n] = bboxes[:, 1:5]
        bbox_valid_mask[:n] = True

    # Add batch dimension:
    # classes: [1, N]
    # coords:  [1, N, 4]
    # mask:    [1, N]
    return (
        bbox_classes.unsqueeze(0),
        bbox_coords.unsqueeze(0),
        bbox_valid_mask.unsqueeze(0)
    )

def get_sample(selected_index=0, dataset_name="NUSCENES", num_frames=25, action_mode="free"):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    # bbox
    bbox_condition = None
    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames
    else:
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]
        bbox_condition = get_bbox_condition(sample_dict)

        path_list = list()
        if dataset_name == "NUSCENES":
            for index in range(num_frames):
                image_path = os.path.join(dataset_dict["data_root"], sample_dict["frames"][index])
                assert os.path.exists(image_path), image_path
                path_list.append(image_path)
            if action_mode != "free":
                action_dict = dict()
                if action_mode == "traj" or action_mode == "trajectory":
                    action_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
                elif action_mode == "cmd" or action_mode == "command":
                    action_dict["command"] = torch.tensor(sample_dict["cmd"])
                elif action_mode == "steer":
                    # scene might be empty
                    if sample_dict["speed"]:
                        action_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
                    # scene might be empty
                    if sample_dict["angle"]:
                        action_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
                elif action_mode == "goal":
                    # point might be invalid
                    if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                        action_dict["goal"] = torch.tensor([
                            sample_dict["goal"][0] / 1600,
                            sample_dict["goal"][1] / 900
                        ])
                else:
                    raise ValueError(f"Unsupported action mode {action_mode}")
        else:
            raise ValueError(f"Invalid dataset {dataset_name}")
    return path_list, selected_index, total_length, action_dict, bbox_condition # bbox


def load_img(file_name, target_height=320, target_width=576, device="cuda"):
    if file_name is not None:
        image = Image.open(file_name)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image file {file_name}")
    ori_w, ori_h = image.size
    # print(f"Loaded input image of size ({ori_w}, {ori_h})")

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])(image)
    return image.to(device)


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()

    if opt.indices_file is not None:
        with open(opt.indices_file, "r") as f:
            eval_indices = [int(line.strip()) for line in f if line.strip()]
    else:
        eval_indices = None
    
    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict, bbox_ckpt=opt.bbox_ckpt)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    if eval_indices is not None:
        eval_pos = 0
        sample_index = eval_indices[eval_pos]
    else:
        sample_index = 0

    while sample_index >= 0:
        if eval_indices is not None:
            print(
                f"Generating subset sample {eval_pos + 1}/{len(eval_indices)} "
                f"(dataset index {sample_index})"
            )

        seed_everything(opt.seed)

        frame_list, sample_index, dataset_length, action_dict, bbox_condition = get_sample(sample_index,
                                                                           opt.dataset,
                                                                           opt.n_frames,
                                                                           opt.action)

        if opt.bbox_ckpt is None:
            bbox_condition = None

        img_seq = list()
        for each_path in frame_list:
            img = load_img(each_path, opt.height, opt.width)
            img_seq.append(img)
        images = torch.stack(img_seq)

        value_dict = init_embedder_options(unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = opt.cond_aug
        value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value

        if opt.n_rounds > 1:
            guider = "TrianglePredictionGuider"
        else:
            guider = "VanillaCFG"
        sampler = init_sampling(guider=guider, steps=opt.n_steps, cfg_scale=opt.cfg_scale, num_frames=opt.n_frames)

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]

        # bbox
        if bbox_condition is not None:
            bbox_classes, bbox_coords, bbox_valid_mask = bbox_condition

            bbox_classes = bbox_classes.cuda()
            bbox_coords = bbox_coords.cuda()
            bbox_valid_mask = bbox_valid_mask.cuda()

            # bbox
            if opt.width == 576 and opt.height == 320:
                bbox_coords = adjust_lowres_bbox_coords(
                    bbox_coords,
                    bbox_valid_mask,
                )
            elif opt.width == 1024 and opt.height == 576:
                # Original bbox coordinates already match this resolution.
                pass
            else:
                raise ValueError(
                    f"Unsupported inference resolution: "
                    f"{opt.width}x{opt.height}"
                )

            print(
                f"BBox inference: "
                f"{bbox_valid_mask.sum().item()} valid objects"
            )
            
        else:
            bbox_classes = None
            bbox_coords = None
            bbox_valid_mask = None
        
        out = do_sample(
            images,
            model,
            sampler,
            value_dict,
            num_rounds=opt.n_rounds,
            num_frames=opt.n_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(opt.n_conds)],
            
            # bbox
            bbox_classes=bbox_classes,
            bbox_coords=bbox_coords,
            bbox_valid_mask=bbox_valid_mask
        )

        if isinstance(out, (tuple, list)):
            samples, samples_z, inputs = out
            virtual_path = os.path.join(opt.save, "virtual")
            real_path = os.path.join(opt.save, "real")
            perform_save_locally(virtual_path, samples, "videos", opt.dataset, sample_index)
            perform_save_locally(virtual_path, samples, "grids", opt.dataset, sample_index)
            perform_save_locally(virtual_path, samples, "images", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "videos", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "grids", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "images", opt.dataset, sample_index)
        else:
            raise TypeError

        if eval_indices is not None:
            eval_pos += 1

            if eval_pos >= len(eval_indices):
                sample_index = -1
            else:
                sample_index = eval_indices[eval_pos]

        elif opt.rand_gen:
            sample_index += random.randint(1, dataset_length - 1)

        else:
            sample_index += 1
            if dataset_length <= sample_index:
                sample_index = -1
