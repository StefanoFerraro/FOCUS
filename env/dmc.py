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
# os.environ["DISPLAY"] = ":0"


object_ids = {"reacher_hard": 3, "reacher_easy": 3, "manipulator_bring_ball": 10}   
limits_exploration_area = {"reacher_hard": [[-0.24, 0.24], [-0.24, 0.24]],
                           "reacher_easy": [[-0.24, 0.24], [-0.24, 0.24]],
                           "manipulator_bring_ball": [[-0.5, 0.5], [0.01, 0.9]]}
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
        # self._env._env._step_limit = self.horizon
        self._seed = seed
        self.task = task
        
        self.segmentation_instances = objs
        self.target_name = objs
        self.gt_segmentation = env_config.renderer.gt_segmentation
        if not self.gt_segmentation:
            self.segmenter = Segmenter(
                env_config,
                self.task,
                self.num_objects,
                img_size=self.seg_size,
                device="cuda:0",
            )
        
        if self.task not in object_ids.keys():
            raise NotImplementedError
        else:
            self.object_id = object_ids[self.task]   
            
        self.limits_exploration_area = env_config.limits_exploration_area = limits_exploration_area[self.task]            

    @property
    def obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self.size, dtype=np.uint8),
            "objects_pos": gym.spaces.Box(
                -2, 2, (len(self.segmentation_instances), 2), dtype=np.float32
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
            # hide target from the scene
            # extension to multiple objects, better to handle it internally the fastSAM method to speed up performance, also need to have a double tracker
            self._env.physics.named.model.geom_rgba[self.target_name, 3] = 0
            high_res_rgb = self._env.physics.render(height=self.seg_size[0], width=self.seg_size[1], camera_id=0)
            self._env.physics.named.model.geom_rgba[self.target_name, 3] = 1
            seg, _, _ = self.segmenter.generate(high_res_rgb, self.is_first)
            seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)

        return proprio, rgb, seg

    def compute_displacements(self, true_objs_pos):
        true_pos_displacement = (
                np.sqrt(np.sum(((true_objs_pos - self.obj_pos) ** 2)))
            )

        return true_pos_displacement

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

    # def get_object_pose(self, seg):
    #     centroid_obj = np.clip(np.mean(np.argwhere(seg), axis=0).astype(int), 0, self.size[1])
    #     return centroid_obj
    
    def get_object_pose(self):        
        obj_pos = self._env.physics.named.data.xpos[self.object_id].copy()
        obj_pos = obj_pos[:2] if "reacher" in self.task else np.array([obj_pos[0], obj_pos[2]])
        # round to 3 decimals, idea to simplify the encoding of the position
        obj_pos = np.around(obj_pos, decimals=3)
        return obj_pos
      
    def step(self, action):
        # assert np.isfinite(action['action']).all(), action['action']
        
        reward = 0.0
        success = 0.0
        self.is_first = False
        time_step = self._env.step(action)
        # env_state, rew, done, info = self._env.step(action)
        success = True if time_step.reward >= 1 else False

        proprio, rgb, seg = self._state_extraction(time_step)

        new_obj_pos = self.get_object_pose()
        
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
            "objects_pos": np.array(objects_pos).astype(np.float32), # reduce the pixel values to decimals
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
        
        if self.task == "manipulator_bring_ball":
            self._env.physics.named.model.geom_rgba["ball"] = [0,1,0,1] # set ball to green for ease of detection                
            # set starting position of the ball to be fixed and not moving.

            _st = self._env.physics.get_state()
            _st[8:10] = [0,0.02]
            self._env.physics.set_state(_st)
            self._env.physics.named.data.qvel[8:10] = 0
        
        proprio, rgb, seg = self._state_extraction(time_step)

        new_obj_pos = self.get_object_pose()
        
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
            "objects_pos": np.array(objects_pos).astype(np.float32), # reduce the pixel values to decimals
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
    
    def set_target(self, target_pos):
        if "reacher" in self.task:
            self._env.physics.named.data.geom_xpos[self.target_name] = [target_pos[0], target_pos[1], 0.01]
            self._env.physics.named.model.geom_pos[self.target_name] = [target_pos[0], target_pos[1], 0.01]
        elif "manipulator" in self.task:
            self._env.physics.named.data.geom_xpos[self.target_name] = [target_pos[0], 0.0, target_pos[1]]
            self._env.physics.named.model.geom_pos[self.target_name] = [target_pos[0], 0.0, target_pos[1]]
        else:
            raise NotImplementedError
           
    def get_rgb_with_target(self, target=None):
        # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
        if "manipulator" in self.task:
            self.set_target(target)
        target_rgb = self._env.physics.render(height=self.size[0], width=self.size[1], camera_id=0).transpose(2, 0, 1)
        return target_rgb
    
    #### ADDED FUNCTIONS ####
    
    # def get_object_pose(self):
    #     self.ee_id = 3 if self.task == "reacher_hard" else NotImplementedError
    #     obj_pos = self._env.physics.named.data.xpos[self.ee_id].copy()
    #     return obj_pos
    
    def get_goals(self):
        return self.goals
  
    def set_goals_for_task(self):
        if self.task in ["reacher_hard", "reacher_easy"]:
            full_right = [[0,0], [0.24, 0]]
            full_left = [[3.14, 0], [-0.24, 0]]
            full_down = [[-1.57, 0], [0, 0.24]]
            full_up = [[1.57, 0], [0, -0.24]]
            top_left_V = [[1.57, 2.], [-0.11, 0.07]]
            down_right_V = [[-1.57, 2.], [0.11, -0.07]]
            top_right_openV = [[0.5, 1], [0.11, 0.18]]
            down_left_openV = [[-2.64, 1], [-0.11, -0.18]]
            close_wrist_right = [[-0.75, 2.5 ], [0.066, 0.036]]
            close_wrist_left = [[2.5, 2.5 ], [-0.062, -0.043]]
        
            self.goals = np.stack([full_right, full_left, full_down, full_up,
                                   top_left_V, down_right_V, top_right_openV,
                                   down_left_openV, close_wrist_right, close_wrist_left])
        
        if self.task in ["manipulator_bring_ball", "manipulator_insert_peg"]:
            full_right = [[-1.57,0,0,0,0.2], [0.53, 0.4]]
            full_left = [[1.57,0,0,0,0.2], [-0.53, 0.4]]
            full_down = [[2.1,1,1,0,0.2], [0, 0.4]]
            full_up = [[0,0,0,0,0.2], [0, 0.91]]
            top_left_V = [[1.8, -1, -1.78, 0, 0.5], [-0.14, 0.59]]
            down_right_V = [[-1.1, -1, -1.78, 0, 0.5], [0.17, 0.24]] 
            top_right_openV = [[0, -1, -0.8, 0, 0.5], [0.32, 0.65]]
            down_left_openV = [[3.14, -1, -0.8, 0, 0.5], [-0.32, 0.15]]
        
            self.goals = np.stack([full_right, full_left, full_down, full_up,
                                   top_left_V, down_right_V, top_right_openV, down_left_openV])
        
        return self.goals
    
    def get_random_goal(self):
        goals = self.set_goals_for_task()
        return goals[np.random.randint(len(goals))]
    
    def render(self):
        return self._env.physics.render(height=self.size[0], width=self.size[0], camera_id=0) #.transpose(2, 0, 1)
    
    def set_goal_state(self, goal):
        size = self._env.physics.get_state().shape[0] - np.array(goal).shape[0]
        self._env.physics.set_state(np.concatenate((goal, np.zeros([size]))))
        self._env.step(np.zeros_like(self.action_space.sample()))
        # self._env.physics.named.data.geom_xpos["target"] = self._env.physics.named.data.xpos[self.ee_id]
        # self._env.physics.named.model.geom_pos["target"] = self._env.physics.named.data.xpos[self.ee_id]
                