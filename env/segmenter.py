import numpy as np
import torch
from ultralytics import YOLO
import sys
from .utils import *
import os
import argparse
import ast
import time
import clip
from env.tracker.base_tracker import BaseTracker


class Tracking:
    def __init__(self, xmem_checkpoint, device="cuda:0"):
        self.xmem_checkpoint = xmem_checkpoint
        self.xmem = BaseTracker(self.xmem_checkpoint, device=device)


class Segmenter:
    def __init__(self, config, num_objects, img_size=[64, 64], device="cuda:0"):
        self.cfg = config

        self.device = device
        self.num_objects = num_objects
        self.img_size = img_size

        # init fast-SAM
        fastSAM_checkpoint = "FastSAM.pt"
        fastSAM_checkpoint_url = config.fastSAM_url
        fastSAM_checkpoint = download_checkpoint(
            fastSAM_checkpoint_url, self.cfg.checkpoints_folder, fastSAM_checkpoint
        )
        self.fastSAM_model = YOLO(fastSAM_checkpoint)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # init xMem https://github.com/hkchengrex/XMem
        xmem_checkpoint = "XMem-s012.pth"
        xmem_checkpoint_url = config.xmem_url

        xmem_checkpoint = download_checkpoint(
            xmem_checkpoint_url, self.cfg.checkpoints_folder, xmem_checkpoint
        )
        self.xmem_model = BaseTracker(xmem_checkpoint, device=self.device)

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
            )
            results = format_results(results[0], 0)
            annotations = np.array(
                self.prompt(image, results, self.cfg, text=True)
            )  # masked results are evalueated and assign a score based on the text prompt

            # inpaint and save image
            # self.fast_process(
            #     annotations=annotations, mask_random_color=self.cfg.randomcolor
            # )

            # starting mask
            template_mask = annotations[0].astype(np.int)

            # add indeces of other masks
            for i, mask in enumerate(annotations[1:]):
                template_mask += (mask * (i + 2)).astype(np.int)

            mask, logit, painted_image = self.xmem_model.track(
                image.copy(), template_mask
            )
        else:
            mask, logit, painted_image = self.xmem_model.track(image.copy())

        return mask, logit, painted_image

    def prompt(self, image, results, args, box=None, point=None, text=None):
        if box:
            mask, _ = self.box_prompt(
                results[0].masks.data,
                convert_box_xywh_to_xyxy(self.cfg.box_prompt),
                self.img_size[0],
                self.img_size[1],
            )
        elif text:
            mask, _ = self.text_prompt(image, results)
        else:
            return None
        return mask

    def text_prompt(self, image, annotations):
        masks = []
        cropped_boxes, cropped_images, not_crop, filter_id, annotaions = crop_image(
            annotations, image
        )
        scores = retriev(
            self.clip_model,
            self.preprocess,
            cropped_boxes,
            self.cfg.text_prompt,
            device=self.device,
        )
        max_idx = scores.argsort()

        for i in range(self.num_objects):
            id = max_idx[-i]  # should get number of objects in the scene
            id += sum(np.array(filter_id) <= int(id))
            masks.append(
                annotaions[id]["segmentation"]
            )  # every object is assigned corresponding id
        return masks, max_idx

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

    # generate image file from mask
    def fast_process(
        self, annotations, mask_random_color, bbox=None, points=None, edgs=False
    ):
        # takes segmented annotations
        if isinstance(annotations[0], dict):
            annotations = [annotation["segmentation"] for annotation in annotations]
        result_name = os.path.basename(self.cfg.img_path)
        # read image
        image = cv2.imread(self.cfg.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if self.cfg.better_quality == True:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                annotations[i] = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )
        if self.cfg.device == "cpu":
            annotations = np.array(annotations)
            fast_show_mask(
                annotations,
                plt.gca(),
                random_color=mask_random_color,
                bbox=bbox,
                points=points,
                pointlabel=self.cfg.point_label,
                retinamask=self.cfg.retina,
                target_height=original_h,
                target_width=original_w,
            )
        else:
            if isinstance(annotations[0], np.ndarray):
                annotations = torch.from_numpy(annotations)
            fast_show_mask_gpu(
                annotations,
                plt.gca(),
                random_color=self.cfg.randomcolor,
                bbox=bbox,
                points=points,
                pointlabel=self.cfg.point_label,
                retinamask=self.cfg.retina,
                target_height=original_h,
                target_width=original_w,
            )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if self.cfg.withContours == True:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask["segmentation"]
                annotation = mask.astype(np.uint8)
                if self.cfg.retina == False:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, hierarchy = cv2.findContours(
                    annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    contour_all.append(contour)
            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        save_path = self.cfg.output
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis("off")
        plt.savefig(save_path + result_name, bbox_inches="tight", pad_inches=0.0)
