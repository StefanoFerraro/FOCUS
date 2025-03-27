from dm_env import specs
import numpy as np
import gym
import os

import robosuite.utils.camera_utils as CU
from env.segmenter import Segmenter

class BaseEnv:
    def __init__(self, env_config, task="", part=["part"], seed=None, action_repeat=1):
        # render parameters
        os.environ["MUJOCO_GL"] = "egl"
        # os.environ["DISPLAY"] = ":0"
        
        # env params
        self.include_background = True
        self.num_objects = len(part)
        self.segmentation_instances = part
    
        self.size = tuple(env_config.renderer.size)
        self.seg_size = tuple(env_config.renderer.seg_size)
        self.reward_shaping = env_config.reward_shaping
        self.task_reward = env_config.task_reward
        self.camera = env_config.renderer.camera
        self.controller = env_config.controller
        self.horizon = env_config.horizon
        self.action_repeat = action_repeat
        self.seed = seed
        self.task = task
        
        self.segmentation_level = env_config.renderer.segmentation_level
        self.gt_segmentation = env_config.renderer.gt_segmentation   
        self.seg_channels = (
            len(self.segmentation_instances) + 1
            if self.include_background
            else len(self.segmentation_instances)
        )
        self.target_part = part
        
        if not self.gt_segmentation:
            self.segmenter = Segmenter(
                env_config,
                self.task,
                img_size=self.seg_size,
                device="cuda:0",
            )
        
        self.cfg = env_config
        
    def obs_specs(self):
        obs_keys = ["rgb", "depth", "proprio", "objects_pos", "segmentation"]
        obs_specs = []
        for k, v in self.obs_space.items():
            if k in obs_keys:
                obs_specs.append(
                    specs.BoundedArray(
                        name=k,
                        shape=v.shape,
                        dtype=v.dtype,
                        minimum=v.low,
                        maximum=v.high,
                    )
                )
        return obs_specs
    
    @property
    def common_obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self.size, dtype=np.uint8),
            "depth": gym.spaces.Box(
                -np.inf,
                np.inf,
                (1,) + self.size,
                dtype=np.float32,
            ),
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
        }
        return spaces
    
    @staticmethod
    def image_segmentation(image, seg):
        """
        Segmentation of image based on segmentation mask, works both with rgb and depth images.
        seg: (dim1, dim2, channels)
        image: (dim1, dim2, 1/3)

        Return:
        seg_image: (dim1, dim2, 1/3, channels)
        """

        seg_chs, _, _ = seg.shape
        image_chs, _, _ = image.shape

        seg_mask = np.repeat(np.expand_dims(seg, axis=1), image_chs, 1)
        seg_image = np.zeros((seg_chs, *image.shape), dtype=image.dtype)

        for ch in range(seg_chs):
            seg_image[ch] = image * seg_mask[ch]

        return seg_image
    
    def seg_background(self, seg, include_background=True):
        if include_background:
            background_mask = np.all(seg == 0, axis=0)
            seg[-1][background_mask] = 1  # last layer is background layer
        return seg
    
    
class ObjectsEnv(BaseEnv):
    def __init__(self, env_config, task="", objs=["obj"], seed=None, action_repeat=1,):
        super().__init__(env_config, task, objs, seed, action_repeat)

        # object specs
        self.cube_rgba = env_config.objects.rgba
        self.cube_minsize = tuple([env_config.objects.minsize] * 3)  # cube
        self.spawn_range = (
            -env_config.objects.spawn_range,
            env_config.objects.spawn_range,
        )
        self.object_name = env_config.objects.name
        self.segmentation_instances = (
            objs
            if task not in ["PickSingleYCB", "CustomLiftYCB"]
            else [self.object_name]
        )
        self.last_estimated_obj_pos = [[0, 0, 0]] * len(
            self.segmentation_instances)  # initialize to zero
        self.objects_pixels = [0] * len(objs)
        self.target_obj = self.segmentation_instances[0]
        self.object_start_pos = env_config.object_start_pos 
        
        self.dist_as_rw = env_config.dist_as_rw

    def pixel_to_world(self, seg, depth):
        depth = depth.transpose(1, 2, 0)

        # depth_map = CU.get_real_depth_map(sim=self._env.sim, depth_map=depth)
        estimated_obj_pos = []

        for ch in range(len(self.segmentation_instances)):
            seg_pixels = np.argwhere(seg[ch])
            if (
                seg_pixels.size > self.objects_pixels[ch]
            ):  # update max number of pixels that compose the objects
                self.objects_pixels[ch] = seg_pixels.size

            if (
                seg_pixels.size > 0.4 * self.objects_pixels[ch]
            ):  # at least 40% of pixels needs to be in view to update the object position
                centroid = np.mean(seg_pixels, axis=0).astype(int)
                estimated_obj_pos += [
                    CU.transform_from_pixels_to_world(
                        pixels=centroid,
                        depth_map=depth,
                        camera_to_world_transform=self.camera_to_world,
                    )
                ]

            else:  # if object is not detected in the scene just take the last relevand position
                estimated_obj_pos += [self.last_estimated_obj_pos[ch]]

        self.last_estimated_obj_pos = estimated_obj_pos

        return estimated_obj_pos

    def is_in_area(self, val, target):
        return abs(target) <= abs(val) <= self.area_threshold and np.sign(
            val
        ) == np.sign(target)

    def min_max_areas(self, val):
        max = self.is_in_area(val, self.area_target)
        min = self.is_in_area(val, -self.area_target)
        return min, max

    def check_in_areas(self, obj_pos):
        left, right = self.min_max_areas(obj_pos[1])
        close, far = self.min_max_areas(obj_pos[0])
        up = (
            self.height_target <= obj_pos[2] - self.height_offset <= self.area_threshold
        )
        return [left, right, close, far, up]
