import os
import cv2
import numpy as np

from dm_env import StepType, specs
import gym

import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import CameraMover as CM
from env.wrappers import PandaGymWrapper

import env.custom_robosuite_tasks as custom_robosuite_tasks

import pyquaternion as pq
from .utils import *
from .base_envs import ObjectsEnv
import mujoco

_LEFT = 0
_RIGHT = 1
_CLOSE = 2
_FAR = 3
_UP = 4

limits_exploration_area = {"Lift": [[-0.25, -0.25, 0], [0.25, 0.25, 0.1]],
                           "Stack": [[-0.25, -0.25, 0], [0.25, 0.25, 0.2]]}

class PandaRoboSuite(ObjectsEnv):
    def __init__(
        self,
        env_config,
        task="Lift",
        objs=["cube"],
        seed=None,
        action_repeat=1,
    ):
        super().__init__(env_config, task, objs, seed, action_repeat)

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

        self._proprio_keys = ["robot0_proprio-state"]
        self._object_keys = ["object-state"]

        self.random_placement = env_config.objects.random_placement

        self.target_x = env_config.goal.x
        self.target_y = env_config.goal.y
        self.target_z = env_config.goal.z
        self.point_goal = env_config.goal.point_goal

        self._obs_keys = [
            self.camera + "_image",
            self.camera + "_depth",
        ]

        self.area_target = 0.25
        self.height_target = 0.05
        self.area_threshold = 0.4
        self.height_offset = 0.8 + self.cube_minsize[2]  # table height + cube height
        
        self.lift_norm = int(abs(1 / (self.area_threshold)) + 1)
        self.push_norm = int(abs(1 / (self.area_threshold - self.area_target)) + 1)
        
        self.limits_exploration_area = env_config.limits_exploration_area = limits_exploration_area[self.task]    
        
        self._make()

    def _make(self):
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

        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()

        self.target_attr_name = self.target_obj
        self.target_obj_attr = getattr(self._env, self.target_attr_name)

    def set_camera_pos(self):
        # move camera closer to robot
        self.cam_mover = CM(self._env, camera=self.camera)
        (
            self.start_cam_pos,
            self.start_cam_quat,
        ) = self.cam_mover.get_camera_pose()
        self.start_cam_pos += self.camera_posmove

        self.cam_mover.set_camera_pose(pos=self.start_cam_pos)

    @property
    def obs_space(self):
        spaces = self.common_obs_space
        spaces.update(
            {
                "proprio": gym.spaces.Box(
                    -5,
                    5,
                    (self._env.modality_dims["robot0_proprio-state"][0],), 
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    -1,
                    1,
                    (3,), # 3 states for the target
                    dtype=np.float32,
                ),
                "state": self._env.observation_space,
            }
        )
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        if self.controller == "OSC_POSE":
            action = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        return {"action": action}

    def obs_specs(self):
        return obs_specs(self.obs_space)

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
        
    
    def get_object_pose(self):
        obj_pos = {}
        obj_ori = {}
        for obj in self.segmentation_instances:
            obj_id = getattr(self._env, obj + "_body_id")
            obj_pos[obj] = self._env.sim.data.body_xpos[obj_id].copy() - self.object_start_pos #remove fixed offset 
            obj_ori[obj] = self._env.sim.data.body_xquat[obj_id].copy()

        return obj_pos.copy(), obj_ori.copy()

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

        seg = self.generate_segmentation(env_state)

        state = {}
        for key in self._proprio_keys or self._obs_keys:
            state[key] = env_state[key]

        return proprio, rgb, depth_map, seg, state

    def generate_segmentation(self, env_state):
        if self.gt_segmentation:
            seg = env_state[self.camera + "_segmentation_" + self.segmentation_level][
                ::-1
            ]
            seg = seg[:,:,0]
        else:
            high_res_rgb = self._env.sim.render(
                camera_name=self.camera, height=self.seg_size[0], width=self.seg_size[1]
            )[::-1]
            seg = self.segmenter.generate(high_res_rgb, self.is_first)
            seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)
        
        return seg
        
    def segmentation_channel_split(self, seg, include_background=False):
        instances_to_ids = self._env.model.instances_to_ids
        seg_map = np.zeros((self.seg_channels, seg.shape[0], seg.shape[1]), dtype=np.uint8)

        if self.gt_segmentation:
            for i, instance in enumerate(self.segmentation_instances):
                for id in instances_to_ids[instance]["geom"]:
                    seg_map[i][seg == id] = 1

            if "Panda0" in self.segmentation_instances:
                panda_idx = self.segmentation_instances.index("Panda0")
                non_panda_idx = [
                    x for x in range(len(self.segmentation_instances)) if x != panda_idx
                ]
                panda_median_layer = cv2.medianBlur(
                    seg_map[panda_idx], 3
                )  # median filtering for panda segmentation

                # check that in panda layer there is no overlap with other encodings
                panda_mask = np.all(seg_map[non_panda_idx] == 0, axis=0)
                seg_map[panda_idx][panda_mask] = panda_median_layer[panda_mask]

        else:

            for i, instance in enumerate(self.segmentation_instances):
                seg_map[i][seg == i + 1] = 1
    
        seg_map = self.seg_background(seg_map, include_background)

        return seg_map
    
    def check_contact(self):
        contact = 0
        for obj in self.segmentation_instances:
            contact += self._env.check_contact(
                self._env.robots[0].gripper, getattr(self._env, obj)
            )
        return contact > 0

    def check_inside_table(self, obj_pos):
        # check if object is inside che edge of the table
        return not (
            abs(obj_pos[0]) > self.area_threshold
            or abs(obj_pos[1]) > self.area_threshold
        )
        
    def push_reward(self, in_areas, true_obj_pos):
        success = in_areas[_RIGHT]  # and done
        reward = (
            (true_obj_pos[1] - self.area_target) * self.push_norm + success
            if success
            else 0
        )
        return reward

    def lift_reward(self, in_areas, true_obj_pos):
        reward_grasp = self._env._check_grasp(
            gripper=self._env.robots[0].gripper,
            object_geoms=self.target_obj_attr,
        )

        success = in_areas[_UP] and reward_grasp
        reward = (
            (true_obj_pos[2] - self.height_offset - self.height_target) * self.lift_norm
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
            true_pos_displacement += (
                np.sqrt(np.sum(((true_objs_pos[obj] - self.true_obj_pos[obj]) ** 2)))
                if self.check_inside_table(
                    true_objs_pos[obj]
                )  # do not consider the vertical dimension in case the object is not inside the table
                else np.sqrt(
                    np.sum(((true_objs_pos[obj][:2] - self.true_obj_pos[obj][:2]) ** 2))
                )
            )

            true_ori_displacement += pq.Quaternion.absolute_distance(
                pq.Quaternion(self.true_obj_ori[obj]),
                pq.Quaternion(true_objs_ori[obj]),
            )
            true_vertical_displacement += (
                abs(true_objs_pos[obj][2] - self.true_obj_pos[obj][2])
                if self.check_inside_table(true_objs_pos[obj])
                else 0
            )

        return (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        )
    
    def step(self, action):
        target_obj = self.segmentation_instances[0]

        if self.controller == "OSC_POSE":
            action = np.insert(action, 3, [0, 0, 0])  # add dummy orientation values

        reward = 0.0
        success = 0.0
        for _ in range(self.action_repeat):
            env_state, rew, done, info = self._env.step(action)
            success = self._env.check_success()
            reward += float(rew)

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

        # objects_pos = self.pixel_to_world(seg, depth)
        objects_pos = list(self.true_obj_pos.values())
        object_to_target = objects_pos[0] - (self._env.env.target_pos - self.object_start_pos)
        if self.dist_as_rw:
            reward = - np.linalg.norm(object_to_target)
        
        action = np.delete(action, [3, 4, 5])  # remove dummy orientation values

        self.is_first = False
        
        if self.cfg.target_ablation_diam:
            rgb = self.get_rgb_with_target()

        obs = {
            "reward": reward,
            "is_first": self.is_first,
            "is_last": done,
            "is_terminal": False,  # will be handled by per_episode function
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "target": np.array(self._env.env.target_pos).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "state": self._env._flatten_obs(state),
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
        self._env.reset()  # reset environment
        reward = 0.0

        self.set_camera_pos()  # move camera closer to the robot
        self.is_first = True

        env_state = self._env._get_observations(force_update=True)

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)
        seg = self.segmentation_channel_split(seg, self.include_background)

        self.true_obj_pos, self.true_obj_ori = self.get_object_pose()

        # objects_pos = self.pixel_to_world(seg, depth)
        objects_pos = list(self.true_obj_pos.values())
        object_to_target = objects_pos[0] - (self._env.env.target_pos - self.object_start_pos)
        if self.dist_as_rw:
            reward = - np.linalg.norm(object_to_target)
        
        if self.cfg.target_ablation_diam:
            rgb = self.get_rgb_with_target()
        
        obs = {
            "reward": reward,
            "is_first": self.is_first,
            "is_last": False,
            "is_terminal": False,
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "target": np.array(self._env.env.target_pos).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "state": self._env._flatten_obs(state),
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

    #### ADDED FUNCTIONS ####

    def _target_hide(self):
        self._env.sim.model.geom("target_g0_vis").rgba[-1] = 0
        
    def _target_show(self):
        self._env.sim.model.geom("target_g0_vis").rgba[-1] = 0.5
        
    def set_target(self, target_pos): 
        # set target in env (visualization purposes)
        self._env.env.target_pos = target_pos + np.array(self.object_start_pos)   
        
    def get_rgb_with_target(self, target=None):
        self._target_show()
        target_rgb = self._env.sim.render(height=self.size[0], width=self.size[1])[::-1].transpose(2, 0, 1)
        self._target_hide()
        return target_rgb
    
    def get_goals(self):
        return self.set_goals_for_task()
  
    def set_goals_for_task(self):
        if self.task in ["Lift"]: # TODO first dimension to define properly
            full_right = [[0.25, 0, 0], [0.25, 0, 0]]
            full_left = [[-0.25, 0, 0], [-0.25, 0, 0]]
            full_down = [[0, 0.25, 0], [0, 0.25, 0]]
            full_up = [[0, -0.25, 0], [0, -0.25, 0]]
            right_down = [[0.12, 0.12, 0], [0.12, 0.12, 0]]
            left_up = [[-0.12, -0.12, 0], [-0.12, -0.12, 0]]
            left_down = [[-0.12, 0.12, 0], [-0.12, 0.12, 0]]
            right_up = [[0.12, -0.12, 0], [0.12, -0.12, 0]]

            self.goals = np.stack([full_right, full_left, full_down, full_up,
                                   right_down, left_up, left_down, right_up])
        else:
            raise NotImplementedError
        return self.goals
    
    def get_random_goal(self):
        goals = self.set_goals_for_task()
        return goals[np.random.randint(len(goals))]
    
    def get_goal(self, index):
        goals = self.set_goals_for_task()
        return goals[index]
    
    def render(self):
        return self._env.sim.render(height=self.size[0], width=self.size[1])[::-1].transpose(2, 0, 1)
    
    def set_goal_state(self, goal_pos):
        goal_pos = goal_pos + np.array(self.object_start_pos)
        goal_pose = np.concatenate([goal_pos, [0, 0, 0, 0]])
        self._env.sim.data.set_joint_qpos(self.segmentation_instances[0] + "_joint0", goal_pose)
    
        return self.step(np.zeros_like(self.act_space["action"].sample()))