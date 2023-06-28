import os
import cv2
import numpy as np

from dm_env import StepType, specs
import gym

import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import CameraMover as CM
from env.wrappers import PandaGymWrapper

import custom_robosuite_tasks

# import pyquaternion as pq

class RealPanda:
    def __init__(self, seg_instances, size):
        self.segmentation_instances = seg_instances
        self.size = size

    def rgb_spec(
        self,
    ):
        v = self.obs_space["rgb"]
        return specs.BoundedArray(
            name="rgb",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def depth_spec(
        self,
    ):
        v = self.obs_space["depth"]
        return specs.BoundedArray(
            name="depth",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def proprio_spec(
        self,
    ):
        v = self.obs_space["proprio"]
        return specs.BoundedArray(
            name="proprio",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def objects_pos_spec(
        self,
    ):
        v = self.obs_space["objects_pos"]
        return specs.BoundedArray(
            name="objects_pos",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def segmentation_spec(
        self,
    ):
        v = self.obs_space["segmentation"]
        return specs.BoundedArray(
            name="segmentation",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def seg_rgb_spec(
        self,
    ):
        v = self.obs_space["seg_rgb"]
        return specs.BoundedArray(
            name="seg_rgb",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def seg_depth_spec(
        self,
    ):
        v = self.obs_space["seg_depth"]
        return specs.BoundedArray(
            name="seg_depth",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def action_spec(
        self,
    ):
        v = self.act_space["action"]
        return specs.BoundedArray(
            name="action",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    @property
    def obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self.size, dtype=np.uint8),
            "depth": gym.spaces.Box(
                0,
                np.inf,
                (1,) + self.size,
                dtype=np.float32,
            ),
            "proprio": gym.spaces.Box(
                -5,
                5,
                (10, ), # 7 proprio + 3 eef =  
                dtype=np.float32,
            ),
            "objects_pos": gym.spaces.Box(
                -2,
                2,
                (self.segmentation_instances, 3),
                dtype=np.float32,
            ),
            "segmentation": gym.spaces.Box(
                0,
                1,
                (self.segmentation_instances + 1,)
                + self.size,
                dtype=np.uint8,
            ),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        return {"action": action}
