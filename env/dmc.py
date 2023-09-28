import os
import cv2
import numpy as np

import gym

import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import CameraMover as CM

import pyquaternion as pq
from .utils import *
from .base_env import BaseEnv
from env.segmenter import Segmenter

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ":0"

class DMCSuiteWrapper():
    def __init__(
        self,
        env,
        task,
        env_config,
        objs=["cube"],
        seed=None,
    ):
        self._env = env
        self.num_objects = len(objs)
        self.include_background = True
        self.size = tuple(env_config.renderer.size)
        self.seg_size = tuple(env_config.renderer.seg_size)
        self.reward_shaping = env_config.reward_shaping
        self.task_reward = env_config.task_reward
        self.camera = env_config.renderer.camera
        self.controller = env_config.controller
        self.horizon = env_config.horizon
        self._seed = seed
        self.task = task
        
        self.segmentation_instances = objs
        self.gt_segmentation = env_config.renderer.gt_segmentation
        if not self.gt_segmentation:
            self.segmenter = Segmenter(
                env_config,
                self.task,
                self.num_objects,
                img_size=self.seg_size,
                device="cuda:0",
            )

    @property
    def obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self.size, dtype=np.uint8),
            "objects_pos": gym.spaces.Box(
                -2, 2, (len(self.segmentation_instances), 3), dtype=np.float32
            ),
            "segmentation": gym.spaces.Box(
                0,
                1,
                (len(self.segmentation_instances) + self.include_background,)
                + self.size,
                dtype=np.uint8,
            ),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "proprio": gym.spaces.Box(
                    -1,
                    1,
                    self._env.observation_spec()["observations"].shape,
                    dtype=np.float32,
                ),
        }
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(
            (spec.minimum) * spec.shape[0],
            (spec.maximum) * spec.shape[0],
            shape=spec.shape,
            dtype=np.float32,
        )
        return {"action": action}

    def _state_extraction(self, env_state):
        rgb = env_state.observation["pixels"].transpose(2, 0, 1)
        proprio = env_state.observation["observations"]

        # obtain world coordinates from the segmentation mask

        if self.gt_segmentation:
            raise NotImplementedError
        else:
            high_res_rgb = self._env.physics.render(height=self.seg_size[0], width=self.seg_size[1])
            seg, _, _ = self.segmenter.generate(high_res_rgb, self.is_first)
            seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)

        return proprio, rgb, seg

    def compute_displacements(self, true_objs_pos):
        true_pos_displacement = 0
        true_ori_displacement = 0


        for obj in self.segmentation_instances:
            true_pos_displacement += (
                np.sqrt(np.sum(((true_objs_pos[obj] - self.obj_pos[obj]) ** 2)))
            )

        return (
            true_pos_displacement,
        )

    def segmentation_channel_split(self, seg, include_background=False):

        channels = (
            len(self.segmentation_instances) + 1
            if include_background
            else len(self.segmentation_instances)
        )

        seg_map = np.zeros(
            (channels, seg.shape[0], seg.shape[1]),
            dtype=np.uint8,
        )

        for i, instance in enumerate(self.segmentation_instances):
            seg_map[i][seg == i + 1] = 1

        if include_background:
            background_mask = np.all(seg_map == 0, axis=0)
            seg_map[-1][background_mask] = 1  # last layer is background layer

        return seg_map

    def get_object_pose(self, seg):
        centroid_obj = np.mean(np.argwhere(seg[0]), axis=0).astype(int)
        return centroid_obj
        
    def step(self, action):
        # assert np.isfinite(action['action']).all(), action['action']
        
        reward = 0.0
        success = 0.0
        self.is_first = False
        time_step = self._env.step(action)
        # env_state, rew, done, info = self._env.step(action)
        success = True if time_step.rew >= 1 else False

        proprio, rgb, seg = self._state_extraction(time_step)

        new_obj_pos = self.get_object_pose(seg)
        
        true_pos_displacement = self.compute_displacements(new_obj_pos)
        # in_areas = self.check_in_areas(new_true_obj_pos)

        reward = (
            max(reward, 0.01) if reward > 0 else 0
        )  # avoid small rewards to help the predictor learning

        self.obj_pos = new_obj_pos

        seg = self.segmentation_channel_split(seg, self.include_background)

        objects_pos = np.array([new_obj_pos])
        
        obs = {
            "reward": time_step.reward,
            "is_first": self.is_first,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "rgb": rgb,
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "success": success,
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": true_pos_displacement,
            "ang_displacement": 0,
            "vertical_displacement": 0,
            "action": action,
            "discount": time_step.discount,
        }
        return obs

    def reset(self):
        self.is_first = True
        
        time_step = self._env.reset()                
        # env_state, rew, done, info = self._env.step(action)

        proprio, rgb, seg = self._state_extraction(time_step)

        new_obj_pos = self.get_object_pose(seg)
        
        self.obj_pos = new_obj_pos

        seg = self.segmentation_channel_split(seg, self.include_background)
        
        objects_pos = np.array([new_obj_pos])
        obs = {
            "reward": 0.0,
            "is_first": self.is_first,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "rgb": rgb,
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "success": False,
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": 0,
            "ang_displacement": 0,
            "vertical_displacement": 0,
            "action": np.zeros_like(self.act_space["action"].sample()),
            "discount": time_step.discount,
        }
        return obs

    def action_spec(self):
        return self._env.action_spec()
    
    def obs_specs(self):
        return obs_specs(self.obs_space)    
    
    # def __getattr__(self, name):
    #     if name == "obs_space":
    #         return self.obs_space
    #     if name == "act_space":
    #         return self.act_space
    #     return getattr(self._env, name)
