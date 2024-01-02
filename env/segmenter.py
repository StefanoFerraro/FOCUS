import numpy as np
import torch
from ultralytics import YOLO
import sys
from .utils import *
import os
import argparse
import ast
import time
from groundingdino.util.inference import load_model, predict, annotate, Model
from torchvision.ops import box_convert

from env.tracker.base_tracker import BaseTracker
from env import MW_TASKS_PROMPT, MS_TASKS_PROMPT, RS_TASKS_PROMPT, DMC_TASKS_PROMPT


class Tracking:
    def __init__(self, xmem_checkpoint, device="cuda:0"):
        self.xmem_checkpoint = xmem_checkpoint
        self.xmem = BaseTracker(self.xmem_checkpoint, device=device)


class Segmenter:
    def __init__(self, config, task, img_size=[64, 64], device="cuda:0"):
        self.cfg = config.segmenter

        self.device = device
        self.img_size = img_size

        seg_prompt = globals()[config.name.upper() + "_TASKS_PROMPT"][task]
        if self.cfg.mode == "None":
            self.cfg.mode = seg_prompt[0]
        self.text_mode = True if self.cfg.mode == "text" else False
        self.box_mode = True if self.cfg.mode == "box" else False

        # init fast-SAM
        fastSAM_checkpoint = "FastSAM.pt"
        fastSAM_checkpoint_url = self.cfg.fastSAM_url
        fastSAM_checkpoint = download_checkpoint(
            fastSAM_checkpoint_url, self.cfg.checkpoints_folder, fastSAM_checkpoint
        )
        self.fastSAM_model = YOLO(fastSAM_checkpoint)

        # self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        GroundingDino_checkpoint = "groundingdino_swint_ogc.pth"
        GroundingDino_checkpoint_url = self.cfg.GroundingDino_url
        GroundingDino_checkpoint = download_checkpoint_wget(
            GroundingDino_checkpoint_url,
            self.cfg.checkpoints_folder,
            GroundingDino_checkpoint,
        )
        groundingdino_config = "configs/env/groundingdino.yaml"
        self.GroundingDino_model = load_model(
            groundingdino_config, GroundingDino_checkpoint
        )

        # init xMem https://github.com/hkchengrex/XMem
        xmem_checkpoint = "XMem-s012.pth"
        xmem_checkpoint_url = self.cfg.xmem_url

        xmem_checkpoint = download_checkpoint(
            xmem_checkpoint_url, self.cfg.checkpoints_folder, xmem_checkpoint
        )
        self.xmem_model = BaseTracker(xmem_checkpoint, device=self.device)

        if self.cfg.text_prompt == "None":
            self.cfg.text_prompt = seg_prompt[1]

        if self.cfg.box_prompt == [0, 0, 0, 0]:
            self.cfg.box_prompt = seg_prompt[1]

    def generate(self, image, is_first):
        if is_first:
            self.xmem_model.clear_memory()

            results = self.fastSAM_model(
                image,
                imgsz=self.img_size,
                device=self.device,
                retina_masks=self.cfg.retina,
                iou=self.cfg.iou,
                conf=self.cfg.conf,
                max_det=100,
                verbose=False,  # stop logging of inference and detection details
            )

            boxes, _, _ = predict(
                model=self.GroundingDino_model,
                image=torch.Tensor(image.copy().transpose(2, 0, 1)).float(),
                caption=self.cfg.text_prompt,
                box_threshold=0.3,
                text_threshold=0.25,
                device=self.device,
            )

            ori_h = image.shape[0]
            ori_w = image.shape[1]

            # Save each frame due to the post process from FastSAM
            boxes = boxes * torch.Tensor([ori_w, ori_h, ori_w, ori_h])
            boxes = (
                box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                .cpu()
                .numpy()
                .tolist()
            )
            masks = []
            for box_idx in range(len(boxes)):
                masks.append(
                    self.box_prompt(
                        results[0].masks.data,
                        boxes[box_idx],
                        ori_h,
                        ori_w,
                    )[0]
                    # .cpu()
                    # .numpy()
                )

            template_mask = np.array(masks).astype(int)
            mask = self.xmem_model.track(
                image.copy(), template_mask[0]
            )

        else:
            mask = self.xmem_model.track(image.copy())

        return mask

    def box_prompt(self, masks, bbox, target_height, target_width):
        h = masks.shape[1]
        w = masks.shape[2]
        if h != target_height or w != target_width:
            bbox = [
                int(bbox[0] * w / target_width),
                int(bbox[1] * h / target_height),
                int(bbox[2] * w / target_width),
                int(bbox[3] * h / target_height),
            ]
        bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
        bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
        bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
        bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        masks_area = torch.sum(
            masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2)
        )
        orig_masks_area = torch.sum(masks, dim=(1, 2))

        union = bbox_area + orig_masks_area - masks_area
        IoUs = masks_area / union
        max_iou_index = torch.argmax(IoUs)

        return masks[max_iou_index].cpu().numpy(), max_iou_index
