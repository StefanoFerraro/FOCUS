import os
import sys

import torch
import numpy as np
from .utils import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "tracker")))

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame

import yaml
from env import MW_TASKS_PROMPT, MS_TASKS_PROMPT, RS_TASKS_PROMPT, DMC_TASKS_PROMPT

def download_models(cfg):
        # TODO check if model download requirement conditional is met
        mobileSAM_checkpoint = "mobileSAM.pt"
        mobileSAM_checkpoint_url = cfg.mobileSAM_url
        GroundingDino_checkpoint = "groundingdino_swint_ogc.pth"
        GroundingDino_checkpoint_url = cfg.GroundingDino_url
        DEVA_checkpoint = "DEVA.pth"
        DEVA_checkpoint_url = cfg.DEVA_url
        
        mobileSAM_checkpoint = download_checkpoint(
            mobileSAM_checkpoint_url, cfg.checkpoints_folder, mobileSAM_checkpoint
        )

        GroundingDino_checkpoint = download_checkpoint_wget(
            GroundingDino_checkpoint_url,
            cfg.checkpoints_folder,
            GroundingDino_checkpoint,
        )

        DEVA_checkpoint = download_checkpoint(
            DEVA_checkpoint_url, cfg.checkpoints_folder, DEVA_checkpoint)
            
        return mobileSAM_checkpoint, GroundingDino_checkpoint, DEVA_checkpoint

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

        if self.cfg.text_prompt == "None":
            self.cfg.text_prompt = seg_prompt[1]

        if self.cfg.box_prompt == [0, 0, 0, 0]:
            self.cfg.box_prompt = seg_prompt[1]
        
        download_models(self.cfg)
        
        self.deva_model = get_model_and_config(self.cfg.DEVA)
        self.gd_model, self.sam_model = get_grounding_dino_model(self.cfg.DEVA, self.device)

        self.deva = DEVAInferenceCore(self.deva_model, config=self.cfg.DEVA) 
        self.deva.next_voting_frame = self.cfg.DEVA['num_voting_frames'] - 1
        self.deva.enabled_long_id()
    
    def generate(self, image, is_first):
        with torch.cuda.amp.autocast(enabled=self.cfg.DEVA.amp):
            with torch.no_grad():
                if is_first:
                    # re-instance deva to clear buffer
                    self.deva = DEVAInferenceCore(self.deva_model, config=self.cfg.DEVA) 
                    # self.deva.clear_buffer()
                prob = process_frame(self.deva, self.gd_model, self.sam_model, is_first, image, self.cfg.text_prompt)

        return prob.detach().to("cpu", dtype=torch.uint8).numpy()

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
