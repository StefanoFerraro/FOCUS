import os
import cv2
import numpy as np

from dm_env import StepType, specs
import gym

import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import CameraMover as CM
from envs.wrappers import PandaGymWrapper

import custom_robosuite_tasks

import pyquaternion as pq

_LEFT = 0
_RIGHT = 1
_CLOSE = 2
_FAR = 3
_UP = 4


class PandaRoboSuite:
    def __init__(
        self,
        env_config,
        task="Lift",
        objs=["cube"],
        seed=None,
        action_repeat=1,
    ):
        os.environ["MUJOCO_GL"] = "egl"

        if env_config.renderer.camera == "agentview1":
            self.camera = "agentview"
            self.camera_posmove = [0.0, 0.0, 0.0]
        elif env_config.renderer.camera == "agentview2":
            self.camera = "agentview"
            self.camera_posmove = [0.2, 0.0, 0.4]
        elif env_config.renderer.camera == "frontview":
            self.camera = "frontview"
            self.camera_posmove = [-0.5, 0.0, 0.0]
        else:
            self.camera = "agentview"
            self.camera_posmove = [0.0, 0.0, 0.0]

        self.size = tuple(env_config.renderer.size)

        self._proprio_keys = ["robot0_proprio-state"]
        # self._proprio_keys = ["robot0_joint_pos_cos"]
        self._object_keys = ["object-state"]
        self.cube_rgba = env_config.objects.rgba
        self.cube_minsize = tuple([env_config.objects.minsize] * 3)  # cube
        self.spawn_range = (
            -env_config.objects.spawn_range,
            env_config.objects.spawn_range,
        )

        self.reward_shaping = env_config.reward_shaping
        self.task_reward = env_config.task_reward

        self.target_x = env_config.goal.x
        self.target_y = env_config.goal.y
        self.target_z = env_config.goal.z
        self.point_goal = env_config.goal.point_goal

        self.objects_pixels = [0] * len(objs)

        self._obs_keys = [
            self.camera + "_image",
            self.camera + "_depth",
        ]
        self.segmentation_instances = objs
        self.include_background = True
        self.segmentation_level = env_config.renderer.segmentation_level
        self.horizon = env_config.horizon

        self._action_repeat = action_repeat
        self._seed = seed
        self.task = task

        self.controller = env_config.controller

        self.area_target = 0.25
        self.height_target = 0.05
        self.area_threshold = 0.4

        self.lift_norm = 3  # min 0 max 0.35 -> normalized 1.05
        self.push_norm = 7  # min 0 max 0.15 -> normalized 1.05

        self.make()

    def get_object_pose(self):
        obj_pos = {}
        obj_ori = {}
        for obj in self.segmentation_instances:
            obj_id = getattr(self._env, obj + "_body_id")
            if self.task == "CustomLift" or self.task == "MoveTo":
                obj_pos[obj] = self._env.sim.data.body_xpos[obj_id].copy()
                obj_ori[obj] = self._env.sim.data.body_xquat[obj_id].copy()
            elif self.task == "CustomStack":
                obj_pos[obj] = self._env.sim.data.body_xpos[obj_id].copy()
                obj_ori[obj] = self._env.sim.data.body_xquat[obj_id].copy()

        return obj_pos.copy(), obj_ori.copy()

    def make(self):
        self._env = PandaGymWrapper(
            custom_robosuite_tasks.make(self.task, self),
            keys=self._proprio_keys + self._obs_keys,
        )

        # move camera closer to robot
        self.set_camera_pos()

        # get camera matrices
        self.world_to_camera = CU.get_camera_transform_matrix(
            sim=self._env.sim,
            camera_name=self.camera,
            camera_height=self.size[0],
            camera_width=self.size[1],
        )
        self.camera_to_world = np.linalg.inv(self.world_to_camera)
        self.last_estimated_obj_pos = [[0, 0, 0]] * len(
            self.segmentation_instances
        )  # initialize to zero
        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()
        self.init_obj_pos = self.true_obj_pos.copy()

    def set_camera_pos(self):

        # move camera closer to robot
        self.cam_mover = CM(self._env, camera=self.camera)
        (
            self.start_cam_pos,
            self.start_cam_quat,
        ) = self.cam_mover.get_camera_pose()
        self.start_cam_pos += self.camera_posmove

        self.cam_mover.set_camera_pose(pos=self.start_cam_pos)

    def segmentation_channel_split(self, seg, include_background=False):
        instances_to_ids = self._env.model.instances_to_ids

        channels = (
            len(self.segmentation_instances) + 1
            if include_background
            else len(self.segmentation_instances)
        )

        seg_map = np.zeros(
            (channels, seg.shape[1], seg.shape[2]),
            dtype=np.uint8,
        )

        for i, instance in enumerate(self.segmentation_instances):
            for id in instances_to_ids[instance]["geom"]:
                seg_map[i][seg[0] == id] = 1

        if "Panda0" in self.segmentation_instances:
            panda_idx = self.segmentation_instances.index("Panda0")
            non_panda_idx = [
                x
                for x in range(len(self.segmentation_instances))
                if x != panda_idx
            ]
            panda_median_layer = cv2.medianBlur(
                seg_map[panda_idx], 3
            )  # median filtering for panda segmentation

            # check that in panda layer there is no overlap with other encodings
            panda_mask = np.all(seg_map[non_panda_idx] == 0, axis=0)
            seg_map[panda_idx][panda_mask] = panda_median_layer[panda_mask]

        if include_background:
            background_mask = np.all(seg_map == 0, axis=0)
            seg_map[-1][background_mask] = 1  # last layer is background layer

        return seg_map

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
        action_space = (
            self._env.action_space
            if self.controller == "JOINT_POSITION"
            else gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        )

        return specs.BoundedArray(
            name="action",
            shape=action_space.shape,
            dtype=action_space.dtype,
            minimum=action_space.low,
            maximum=action_space.high,
        )

    @property
    def obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self.size, dtype=np.uint8),
            "depth": gym.spaces.Box(
                -np.inf,
                np.inf,
                (1,) + self.size,
                dtype=np.float32,
            ),
            "proprio": gym.spaces.Box(
                -5,
                5,
                self._env.modality_dims["robot0_proprio-state"],
                dtype=np.float32,
            ),
            "objects_pos": gym.spaces.Box(
                -2,
                2,
                (len(self.segmentation_instances), 3),
                dtype=np.float32,
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
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        if self.controller == "OSC_POSE":
            action = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        return {"action": action}

    def _proprio_obs(self, state):
        proprio = []

        for key in self._proprio_keys:
            proprio += list(state[key])

        return proprio

    def _state_extraction(self, env_state):
        proprio = self._proprio_obs(env_state)

        rgb = env_state[self.camera + "_image"][::-1].transpose(2, 0, 1)
        depth = env_state[self.camera + "_depth"][::-1].transpose(2, 0, 1)

        # obtain world coordinates from the segmentation mask
        depth_map = CU.get_real_depth_map(sim=self._env.sim, depth_map=depth)

        seg = env_state[
            self.camera + "_segmentation_" + self.segmentation_level
        ][::-1].transpose(2, 0, 1)

        state = {}
        for key in self._proprio_keys or self._obs_keys:
            state[key] = env_state[key]

        return proprio, rgb, depth_map, seg, state

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

    def check_contact(self):
        contact = 0
        for obj in self.segmentation_instances:
            contact += self._env.check_contact(
                self._env.robots[0].gripper, getattr(self._env, obj)
            )
        return contact > 0

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
            self.height_target
            <= obj_pos[2]
            - self.init_obj_pos[self.segmentation_instances[0]][2]
            <= self.area_threshold
        )
        return [left, right, close, far, up]

    def step(self, action):

        target_obj = self.segmentation_instances[0]

        if self.controller == "OSC_POSE":
            action = np.insert(
                action, 3, [0, 0, 0]
            )  # add dummy orientation values

        # assert np.isfinite(action["action"]).all(), action["action"]

        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            env_state, rew, done, info = self._env.step(action)
            success = self._env.check_success()
            reward += float(rew)
        # success = (
        #     min(success, 1.0) and done
        # )  # success only assigned at last step

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)

        contact = self.check_contact()

        # get objects position and orientation
        new_true_obj_pos, new_true_obj_ori = self.get_object_pose()

        true_pos_displacement = 0
        true_ori_displacement = 0
        for obj in self.segmentation_instances:
            true_pos_displacement += np.sqrt(
                np.sum(((new_true_obj_pos[obj] - self.true_obj_pos[obj]) ** 2))
            )

            true_ori_displacement += pq.Quaternion.absolute_distance(
                pq.Quaternion(self.true_obj_ori[obj]),
                pq.Quaternion(new_true_obj_ori[obj]),
            )

        in_areas = self.check_in_areas(new_true_obj_pos[target_obj])
        if self.task_reward == "lift":
            success = in_areas[_UP]  # and done
            reward = (
                (new_true_obj_pos[target_obj][2] - self.height_target)
                * self.lift_norm
                if success
                else 0
            )
        elif self.task_reward == "push":
            success = in_areas[_RIGHT]  # and done
            reward = (
                (new_true_obj_pos[target_obj][1] - self.area_target)
                * self.push_norm
                if success
                else 0
            )
        else:
            # do not update reward or success
            pass

        self.true_obj_pos = new_true_obj_pos
        self.true_obj_ori = new_true_obj_ori

        seg = self.segmentation_channel_split(seg, self.include_background)

        objects_pos = self.pixel_to_world(seg, depth)
        action = np.delete(
            action, [3, 4, 5]
        )  # remove dummy orientation values

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": done,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "state": self._env._flatten_obs(state),
            "action": action,
            "success": bool(success),
            "in_areas": in_areas,
            "contact": contact,
            "pos_displacement": true_pos_displacement,
            "ang_displacement": true_ori_displacement,
            "discount": 1,
        }

        return obs

    def reset(self):

        self._env.reset()  # reset environment

        self.set_camera_pos()  # move camera closer to the robot

        env_state = self._env._get_observations(force_update=True)

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)
        seg = self.segmentation_channel_split(seg, self.include_background)

        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()

        objects_pos = self.pixel_to_world(seg, depth)

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "state": self._env._flatten_obs(state),
            "action": np.zeros_like(self.act_space["action"].sample()),
            "success": False,
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": 0,
            "ang_displacement": 0,
            "discount": 1,
        }

        return obs
