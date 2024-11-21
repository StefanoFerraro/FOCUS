import os

import numpy as np
from dm_env import StepType, specs

import gym
from gym import spaces

import robosuite.utils.camera_utils as CU

import mani_skill2.envs
import env.custom_maniskill_tasks as custom_maniskill_tasks

from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper
from mani_skill2.utils.common import (
    flatten_dict_space_keys,
    flatten_state_dict,
)
from mani_skill2.utils.sapien_utils import (
    get_pairwise_contacts,
    get_articulation_contacts,
)

from sapien.core import Pose
import pyquaternion as pq
from mani_skill2.utils.wrappers import RecordEpisode
from env.utils import *
from env.base_envs import ObjectsEnv

_LEFT = 0
_RIGHT = 1
_CLOSE = 2
_FAR = 3
_UP = 4


class PandaManiSkill(ObjectsEnv):
    def __init__(
        self,
        env_config,
        task="Stack",
        objs=["obj"],
        seed=None,
        action_repeat=1,
    ):
        super().__init__(env_config, task, objs, seed, action_repeat)

        non_cube_envs = ["TurnFaucet", "PickSingleYCB", "CustomLiftYCB"]
        self.env_id = task + "Cube-v0" if task not in non_cube_envs else task + "-v0"
        self.obs_mode = (
            "state_dict"  # ['image', 'pointcloud', 'rgbd', 'state_dict', 'state']
        )
        self.reward_mode = (
            "dense" if env_config.reward_shaping else "sparse"
        )  # @param can be one of ['sparse', 'dense']

        self._proprio_keys = {"extra": ["tcp_pose"], "agent": ["qpos", "qvel"]}
        self._object_keys = {"extra": [obj + "_pose" for obj in objs]}

        self.target_x = env_config.goal.x
        self.target_y = env_config.goal.y
        self.target_z = env_config.goal.z
        self.point_goal = env_config.goal.point_goal

        self.objects_pixels = [0] * len(objs)
        self._obs_keys = {"base_camera": ["Color", "Position", "Segmentation"]}

        self.area_target = 0.4
        self.height_target = 0.1
        self.area_threshold = 0.5

        self.lift_norm = int(
            abs(1 / (self.area_threshold - self.height_target) + 1)
        )  # min 0 max 0.35 -> normalized 1.05
        self.push_norm = int(
            abs(1 / (self.area_threshold - self.area_target) + 1)
        )  # min 0 max 0.15 -> normalized 1.05

        self.make()

    def _filter_obs(self, obs, keys):
        out_obs = {}
        for key, values in list(obs.items()):
            if key in list(keys):
                for value in list(values):
                    if value in keys[key]:
                        out_obs[value] = obs[key][value]
        return out_obs

    def _keys_rename(self, obs):
        obs["rgb"] = obs.pop("Color")
        obs["depth"] = obs.pop("Position")
        obs.pop("Segmentation")  # remove segmentation from dict, not in the state dict

        obs["proprio"] = np.array(
            (list(obs.pop("qpos")) + list(obs.pop("qvel")) + list(obs.pop("tcp_pose"))),
            dtype=np.float32,
        )

        return obs

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observationsself.target_obj flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def get_object_pose(self):
        obj_pos = {}
        obj_ori = {}

        for obj in self.segmentation_instances:
            if self.task in ["Lift", "MoveTo", "PickSingleYCB", "CustomLiftYCB"]:
                obj_pos[obj] = self._env.unwrapped.obj.get_pose().p
                obj_ori[obj] = self._env.unwrapped.obj.get_pose().q
            elif self.task == "Stack":
                obj_attr = getattr(self._env.unwrapped, obj)
                obj_pos[obj] = obj_attr.get_pose().p
                obj_ori[obj] = obj_attr.get_pose().q
            elif self.task == "TurnFaucet":
                obj_pos[obj] = self._env.unwrapped.faucet.get_pose().p
                obj_ori[obj] = self._env.unwrapped.faucet.get_qpos()[0]
        return obj_pos, obj_ori

    def make(self):
        self._env = custom_maniskill_tasks.make(self.env_id, self)

        self._env = ContinuousTaskWrapper(self._env, self.horizon)
        if self.task == "TurnFaucet":
            cameras = self._env.unwrapped._scene.get_cameras()
            for cam in cameras:
                if cam.name == self.camera:
                    cam.set_pose(
                        Pose(
                            [0.3, 0, 0.6],
                            [6.50068e-17, 0.433189, -1.11022e-16, -0.901303],
                        )
                    )

        obs = self._env.reset()
        self._env.render(mode="cameras")
        images = self._env.get_images()
        self.target_object_actor_ids = (
            [
                x.id
                for x in self._env.get_actors()
                if x.name not in ["ground", "goal_site"]
            ]
            if self.task != "TurnFaucet"
            else [self._env.unwrapped.target_link.id]
        )

        self.target_attr_name = (
            self.target_obj
            if self.task
            not in ["MoveTo", "CustomLift", "PickSingleYCB", "CustomLiftYCB"]
            else "obj"
        )
        self.target_obj_attr = getattr(self._env.unwrapped, self.target_attr_name)

        obs = {**obs, **images}

        self.camera_params = self._env.get_camera_params()
        self.intr = self.camera_params["base_camera"]["intrinsic_cv"]
        self.intr = np.c_[self.intr, np.zeros(3)]

        self.extr = self.camera_params["base_camera"]["extrinsic_cv"]

        self.world_to_camera = np.append(
            np.matmul(self.intr, self.extr), [[0, 0, 0, 1]], axis=0
        )
        self.camera_to_world = np.linalg.inv(self.world_to_camera)

        self.last_estimated_obj_pos = [[0, 0, 0]] * len(
            self.segmentation_instances
        )  # initialize to zero

        self.dict_keys = {**self._proprio_keys, **self._obs_keys}
        self.keys = ["rgb", "depth", "proprio"]

        # filter state based on keys
        obs = self._keys_rename(self._filter_obs(obs, self.dict_keys))

        # Concatenate all the states
        self.modality_dims = {key: obs[key].shape for key in self.keys}

        state = self._flatten_obs(obs)
        self.obs_dim = state.shape
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        self.action_space = self._env.action_space

        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()
        self.init_obj_pos = self.true_obj_pos.copy()
        self.height_offset = self.init_obj_pos[self.segmentation_instances[0]][2]

    def segmentation_channel_split(self, seg, include_background=False):
        channels = (
            len(self.segmentation_instances) + 1
            if include_background
            else len(self.segmentation_instances)
        )

        seg_map = np.zeros(
            (channels, seg.shape[1], seg.shape[2]),
            dtype=np.uint8,
        )

        for i, instance in enumerate(self.target_object_actor_ids):
            seg_map[i][seg[0] == instance] = 1

        if include_background:
            background_mask = np.all(seg_map == 0, axis=0)
            seg_map[-1][background_mask] = 1  # last layer is background layer

        return seg_map

    @property
    def obs_space(self):
        spaces = self.common_obs_space
        spaces.update(
            {
                "proprio": gym.spaces.Box(
                    -5,
                    5,
                    self.modality_dims["proprio"],
                    dtype=np.float32,
                ),
                "state": gym.spaces.Box(
                    -np.inf, np.inf, (self.obs_dim), dtype=np.float32
                ),
            }
        )
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def obs_specs(self):
        return obs_specs(self.obs_space)

    def action_spec(
        self,
    ):
        return specs.BoundedArray(
            name="action",
            shape=self._env.action_space.shape,
            dtype=self._env.action_space.dtype,
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
        )

    def _state_extraction(self, env_state):
        # concatenate proprio and obs
        self._env.render(mode="cameras")
        images = self._env.get_images()
        state = {**env_state, **images}
        state_dict = self._keys_rename(self._filter_obs(state, self.dict_keys))
        rgb = state_dict["rgb"][..., :3]
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8).transpose(2, 0, 1)

        if self.gt_segmentation:
            seg = np.expand_dims(state[self.camera]["Segmentation"][:, :, 1], axis=0)
        else:
            seg = self.segmenter.generate(rgb, self.is_first)

        proprio = state_dict["proprio"]

        depth_map = -state_dict["depth"][..., [2]].transpose(2, 0, 1)

        return proprio, rgb, depth_map, seg, state_dict

    def check_contact(self):
        contacts = self._env.unwrapped._scene.get_contacts()
        excluded_actor_ids = [
            x
            for x in self._env.get_actors()
            if x.name not in self.segmentation_instances
        ]
        objects_actor_ids = (
            [x for x in self._env.get_actors() if x.name in self.segmentation_instances]
            if self.task != "TurnFaucet"
            else [self._env.unwrapped.target_link]
        )

        for contact in contacts:
            if (
                contact.actor0 in objects_actor_ids
            ):  # contact needs to be with the objects
                if contact.actor1 not in excluded_actor_ids + objects_actor_ids:
                    return True

            elif contact.actor1 in objects_actor_ids:
                if contact.actor0 not in excluded_actor_ids + objects_actor_ids:
                    return True

        return False

    def push_reward(self, in_areas, true_obj_pos):
        success = in_areas[_RIGHT]
        reward = (
            (true_obj_pos[1] - self.area_target) * self.push_norm + success
            if success
            else 0
        )
        return reward

    def lift_reward(self, in_areas, true_obj_pos):
        reward_grasp = self._env.unwrapped.agent.check_grasp(self.target_obj_attr)
        success = in_areas[_UP] and reward_grasp
        reward = (
            max(
                (true_obj_pos[2] - self.height_target - self.height_offset)
                * self.lift_norm,
                0,
            )
            + reward_grasp
            if success
            else 0
        )
        return reward

    def compute_displacements(self, true_objs_pos, true_objs_ori):
        true_pos_displacement = 0
        true_ori_displacement = 0
        true_vertical_displacement = 0

        for obj in self.segmentation_instances:
            true_pos_displacement += np.sqrt(
                np.sum(((true_objs_pos[obj] - self.true_obj_pos[obj]) ** 2))
            )

            true_ori_displacement += (
                pq.Quaternion.absolute_distance(
                    pq.Quaternion(self.true_obj_ori[obj]),
                    pq.Quaternion(true_objs_ori[obj]),
                )
                if self.task != "TurnFaucet"
                else abs(true_objs_ori[obj] - self.true_obj_ori[obj])
            )
            true_vertical_displacement += abs(
                true_objs_pos[obj][2] - self.true_obj_pos[obj][2]
            )

        return (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        )

    def step(self, action):
        # assert np.isfinite(action["action"]).all(), action["action"]
        target_obj = self.segmentation_instances[0]
        reward = 0.0
        success = 0.0
        self.is_first = False
        for _ in range(self.action_repeat):
            env_state, rew, done, info = self._env.step(action)
            reward += float(rew)
            success = float(info["success"])
        proprio, rgb, depth, seg, state = self._state_extraction(env_state)

        contact = self.check_contact()

        new_true_obj_pos, new_true_obj_ori = self.get_object_pose()
        (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        ) = self.compute_displacements(new_true_obj_pos, new_true_obj_ori)

        in_areas = self.check_in_areas(new_true_obj_pos[target_obj])

        if self.task_reward == "lift":
            reward = self.lift_reward(in_areas, new_true_obj_pos[target_obj])
        elif self.task_reward == "push":
            reward = self.push_reward(in_areas, new_true_obj_pos[target_obj])
        else:
            # Do not change success or reward, use original
            pass

        reward = (
            max(reward, 0.01) if reward > 0 else 0
        )  # avoid small rewards to help the predictor learning

        self.true_obj_pos = new_true_obj_pos
        self.true_obj_ori = new_true_obj_ori

        seg = self.segmentation_channel_split(seg, self.include_background)

        objects_pos = self.pixel_to_world(seg, depth)

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
            "state": self._flatten_obs(state),
            "action": action,
            "success": bool(success),
            "in_areas": in_areas,
            "contact": contact,
            "pos_displacement": true_pos_displacement,
            "ang_displacement": true_ori_displacement,
            "vertical_displacement": true_vertical_displacement,
            "discount": 1,
        }

        return obs

    def reset(self):
        env_state = self._env.reset()  # reset environment
        self.is_first = True

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)
        seg = self.segmentation_channel_split(seg, self.include_background)

        objects_pos = self.pixel_to_world(seg, depth)

        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()

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
            "state": self._flatten_obs(state),
            "action": np.zeros_like(self.act_space["action"].sample()),
            "success": False,
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": 0,
            "ang_displacement": 0,
            "vertical_displacement": 0,
            "discount": 1,
        }

        return obs
