import os
import cv2
import numpy as np

import gym

import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import CameraMover as CM

import pyquaternion as pq
from .utils import *
from .base_envs import BaseEnv
from env.segmenter import Segmenter

from env.wrappers import *
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_control.utils import rewards

import env.custom_dmc_tasks as cdmc


part_ids = {"reacher_hard": 3, "reacher_easy": 3, "manipulator_bring_ball": 10}   
part_bodyids = {"reacher_hard": 9, "reacher_easy": 9}   

limits_exploration_area = {"reacher_hard": [[-0.24, -0.24], [0.24, 0.24]],
                           "reacher_easy": [[-0.24, -0.24], [0.24, 0.24]],
                           "manipulator_bring_ball": [[-0.5, 0.01], [0.5, 0.9]]}

class DMCSuite(BaseEnv):
    def __init__(
        self,
        env_config,        
        task="recher_hard",
        part="hand",
        seed=0,
        action_repeat=1,
    ):
        
        super().__init__(env_config, task, part, seed, action_repeat)
        
        if self.task not in part_ids.keys():
            raise NotImplementedError
        else:
            self.part_id = part_ids[self.task]   
            self.part_bodyid = part_bodyids[self.task]
                
        self.limits_exploration_area = env_config.limits_exploration_area = limits_exploration_area[self.task]    
        self.visualize_target = env_config.visualize_target
        
        self._make()
                
    def _make(self, ):
        visualize_reward = False
        subdomain, subtask = self.task.split("_", 1)
        
        # if task is not in the suite, use custom tasks
        if (subdomain, subtask) in suite.ALL_TASKS:
            self._env = suite.load(
                subdomain,
                subtask,
                task_kwargs=dict(random=self.seed, time_limit = self.horizon * 0.02), # 0.02 is the timestep length
                environment_kwargs=dict(flat_observation=True),
                visualize_reward=visualize_reward,
            )
        else:
            self._env = cdmc.make(
                subdomain,
                subtask,
                task_kwargs=dict(random=self.seed),
                environment_kwargs=dict(flat_observation=True),
                visualize_reward=visualize_reward,
            )
        
        self._env = ActionDTypeWrapper(self._env, np.float32)
        self._env = ActionRepeatWrapper(self._env, self.action_repeat)
        
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(subdomain, 0)
        
        self._env._camera = camera_id
        render_kwargs = dict(height=self.size[0], width=self.size[1], camera_id=camera_id)
        self._env = pixels.Wrapper(self._env, pixels_only=False, render_kwargs=render_kwargs)
        
        self._env = action_scale.Wrapper(self._env, minimum=-1.0, maximum=+1.0)
        self._env = ExtendedTimeStepWrapper(self._env)
    
    @property
    def obs_space(self):
        spaces = self.common_obs_space
        del spaces["depth"] # depth not given in dmc env
        spaces.update(
            {
                "proprio": # self._env.observation_spec()["observations"],  
                    gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    self._env.observation_spec()["observations"].shape,
                    dtype=np.float32,
                ),
                "objects_pos": gym.spaces.Box(-2, 2, (len(self.segmentation_instances), 2), dtype=np.float32
                ),
                # "dist_to_target": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            }
        )
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

    def action_spec(self):
        return self._env.action_spec()
    
    def obs_specs(self):
        return obs_specs(self.obs_space)    
    
    def get_object_pose(self):        
        obj_pos = self._env.physics.named.data.xpos[self.part_id].copy()
        obj_pos = obj_pos[:2] if "reacher" in self.task else np.array([obj_pos[0], obj_pos[2]])
        # round to 3 decimals, idea to simplify the encoding of the position
        obj_pos = np.around(obj_pos, decimals=3)
        return obj_pos
    
    def _state_extraction(self, env_state):
        rgb = env_state.observation["pixels"].transpose(2, 0, 1)
        proprio = env_state.observation["observations"]
        seg = self.generate_segmentation()
        
        return proprio, rgb, seg
    
    def generate_segmentation(self): 
        if self.gt_segmentation:
            seg = self._env.physics.render(height=self.size[0], width=self.size[0], camera_id=0, segmentation=True)
            seg = seg[:,:,0]
        else:  
            self._env.physics.named.model.geom_rgba[self.target_part, 3] = 0
            high_res_rgb = self._env.physics.render(height=self.seg_size[0], width=self.seg_size[1], camera_id=0)
            self._env.physics.named.model.geom_rgba[self.target_part, 3] = 1
            seg = self.segmenter.generate(high_res_rgb, self.is_first)
            seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)

        return seg
    
    def segmentation_channel_split(self, seg, include_background=False):
        
        seg_map = np.zeros((self.seg_channels, seg.shape[0], seg.shape[1]), dtype=np.uint8)

        if self.gt_segmentation:
            for i, _ in enumerate(self.segmentation_instances):
                seg_map[i][seg == self.part_bodyid] = 1
        else:
            for i, _ in enumerate(self.segmentation_instances):
                seg_map[i][seg == i + 1] = 1

        seg_map = self.seg_background(seg_map, include_background)

        return seg_map
    
    def compute_displacements(self, true_objs_pos):
        true_pos_displacement = (
                np.sqrt(np.sum(((true_objs_pos - self.obj_pos) ** 2)))
            )

        return true_pos_displacement
      
    def step(self, action):
        # assert np.isfinite(action['action']).all(), action['action']
        
        reward = 0.0
        success = 0.0
        self.is_first = False
        time_step = self._env.step(action)
        # env_state, rew, done, info = self._env.step(action)
        reward = time_step.reward
        reward = (
            max(reward, 0.01) if reward > 0 else 0
        )  # avoid small rewards to help the predictor learning
        success = True if reward >= 1 else False
        
        proprio, rgb, seg = self._state_extraction(time_step)
        new_obj_pos = self.get_object_pose()
        if self.dist_as_rw:
            reward = - np.sqrt(np.sum(proprio[2:4]**2))
        
        true_pos_displacement = self.compute_displacements(new_obj_pos)
        # in_areas = self.check_in_areas(new_true_obj_pos)

        self.obj_pos = new_obj_pos

        seg = self.segmentation_channel_split(seg, self.include_background)

        objects_pos = np.array([new_obj_pos])
        
        obs = {
            "reward": reward,
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
        
        if self.cfg.target_ablation_diam:
            obs = custom_target_2d(obs, self._env.physics.named.data.geom_xpos[self.target_part][0][:2], self.cfg.target_ablation_diam, self)
        
        return obs

    def reset(self):
        self.is_first = True
        reward = 0.0
        if not self.visualize_target:
            self._env.physics.named.model.geom_rgba["target"][-1] = 0 # hide the target          
             
        time_step = self._env.reset() 
        
        if self.task == "manipulator_bring_ball":
            self._env.physics.named.model.geom_rgba["ball"] = [0,1,0,1] # set ball to green for ease of detection                
            # set starting position of the ball to be fixed and not moving.

            _st = self._env.physics.get_state()
            _st[8:10] = [0,0.02]
            self._env.physics.set_state(_st)
            self._env.physics.named.data.qvel[8:10] = 0
        
        
        proprio, rgb, seg = self._state_extraction(time_step)
        if self.dist_as_rw:
            reward = - np.sqrt(np.sum(proprio[2:4]**2))
        
        new_obj_pos = self.get_object_pose()
        
        self.obj_pos = new_obj_pos

        seg = self.segmentation_channel_split(seg, self.include_background)
        
        objects_pos = np.array([new_obj_pos])
        obs = {
            "reward": reward,
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
        
        if self.cfg.target_ablation_diam:
            obs = custom_target_2d(obs, self._env.physics.named.data.geom_xpos[self.target_part][0][:2], self.cfg.target_ablation_diam, self)
        
        return obs
    
    #### ADDED FUNCTIONS ####
    
    def set_target(self, target_pos):
        if "reacher" in self.task:
            self._env.physics.named.data.geom_xpos[self.target_part] = [target_pos[0], target_pos[1], 0.01]
            self._env.physics.named.model.geom_pos[self.target_part] = [target_pos[0], target_pos[1], 0.01]
        elif "manipulator" in self.task:
            self._env.physics.named.data.geom_xpos[self.target_part] = [target_pos[0], 0.0, target_pos[1]]
            self._env.physics.named.model.geom_pos[self.target_part] = [target_pos[0], 0.0, target_pos[1]]
        else:
            raise NotImplementedError
    
    def get_target(self):
        if "reacher" in self.task:
            target = self._env.physics.named.data.geom_xpos[self.target_part][0, :2]
        elif "manipulator" in self.task:
            target = self._env.physics.named.data.geom_xpos[self.target_part][0, 0::2]
        else:
            raise NotImplementedError
        return target
    
    def get_rgb_with_target(self, target=None):
        # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
        self._env.physics.named.model.geom_rgba["target"][-1] = 1 # be sure that the target is in view          
        if "manipulator" in self.task:
            self.set_target(target)
        target_rgb = self._env.physics.render(height=self.size[0], width=self.size[1], camera_id=0).transpose(2, 0, 1)

        if not self.visualize_target: self._env.physics.named.model.geom_rgba["target"][-1] = 0 # hide the target 
        return target_rgb
    
    def get_goals(self):
        return self.set_goals_for_task()
  
    def set_goals_for_task(self):
        if self.task in ["reacher_hard", "reacher_easy"]:
            full_right = [[0,0], [0.24, 0]]
            full_left = [[3.14, 0], [-0.24, 0]]
            full_down = [[-1.57, 0], [0, -0.24]]
            full_up = [[1.57, 0], [0, 0.24]]
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
    
    def get_goal(self, index):
        goals = self.set_goals_for_task()
        return goals[index]
    
    def render(self):
        return self._env.physics.render(height=self.size[0], width=self.size[0], camera_id=0) #.transpose(2, 0, 1)
    
    def set_goal_state(self, goal):
        size = self._env.physics.get_state().shape[0] - np.array(goal).shape[0]
        self._env.physics.set_state(np.concatenate((goal, np.zeros([size]))))
        return self.step(np.zeros_like(self.act_space["action"].sample()))

    def get_reward(self, dist):
        radii = 0.06
        return rewards.tolerance(dist, (0, radii))