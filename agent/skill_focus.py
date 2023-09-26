import torch.nn as nn
import torch
from torch.nn.functional import normalize

import utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np

from agent.focus import FocusAgent
from agent.skill_utils import SkillActorCritic

def stop_gradient(x):
    return x.detach()

Module = nn.Module

class SkillFocusAgent(FocusAgent):
    def __init__(self, name, cfg, obs_space, act_spec, is_finetune, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, is_finetune, **kwargs)
        
        self.exploration_area = self.cfg.env.init_exploration_area
        # sample a circle from the center of the workspace
        self._rad_circle = self.exploration_area[0]
        self.fixed_height = self.exploration_area[1]
        
        self.update_target()
        
        # NOTE: Only for debugging
        self._skill_strategy = 'object_context_pose'
        self._fixed_skill = self.wm.heads["object_decoder"](poses=stop_gradient(self._target_pos))["prior"][0][0]
        self.skill_dim = [self._fixed_skill.shape[-1]]

        self._skill_behavior = SkillActorCritic(cfg, self.act_spec, self.tfstep, skill_dim=self.skill_dim, 
                                                    sampling_strategy=self._skill_strategy, solved_meta={'skill' : self._fixed_skill}) 

        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)

    def update_target(self):
        new_target = np.random.uniform([-self._rad_circle] * 2, [self._rad_circle] * 2)
        new_target= np.append(new_target, self.fixed_height) # z dimension fixed  
        self._target_pos = torch.Tensor([[[new_target]]]).to(device="cuda") 
        
    def set_radius_target(self, rad):
        self._rad_circle = rad
    
    def object_context_position_reward_fn(self, seq):
        obj_id, obj_goal_pos = torch.split(seq['skill'] , self.skill_dim, dim=-1) 
        
        obj_poses = self.wm.heads["object_decoder"](seq["feat"], only_mlp=True)[
            "objects_pos"
        ].mean
        T, B, O, P = obj_poses.shape
        obj_pos = obj_poses.reshape(T*B, O, P)[torch.arange(T*B), torch.argmax(obj_id,-1).reshape(T*B)].reshape(T,B,P)
        squared_distance = torch.sum(((obj_goal_pos - obj_pos) ** 2), dim=2).unsqueeze(-1) 
        return -squared_distance

    def object_context_pose_reward_fn(self, seq):
        # obj_id = torch.eye(2)
        post_obj_state = self.wm.heads["object_decoder"](seq["feat"])["post"][:,:,0,:] #consider only first object
        # T, B, O, P = post_obj_state.shape
        # post_obj_state = post_obj_state.reshape(T*B, O, P)[torch.arange(T*B), torch.argmax(obj_id,-1).reshape(T*B)].reshape(T,B,P)
        squared_distance = torch.sum(((post_obj_state - self._fixed_skill) ** 2), dim=2).unsqueeze(-1) 
        return -squared_distance

    def update(self, data, step):
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        start = {k: stop_gradient(v) for k, v in start.items()}

        self._fixed_skill = self.wm.heads["object_decoder"](poses=stop_gradient(self._target_pos))["prior"][0][0]
        self._skill_behavior.solved_meta['skill'] = self._fixed_skill
        
        # update based on mode, save compute time
        metrics.update(
            self._skill_behavior.update(
                self.wm, start, data["is_terminal"], getattr(self, f'{self._skill_strategy}_reward_fn')
            )
        )
        return state, metrics

    def act(self, obs, meta, step, eval_mode, state):
        obs = {
            k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0)
            for k, v in obs.items()
        }
        if state is None:
            latent = self.wm.rssm.initial(len(obs["reward"]))
            action = torch.zeros(
                (len(obs["reward"]),) + self.act_spec.shape, device=self.device
            )
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], should_sample
        )
        feat = self.wm.rssm.get_feat(latent)

        # only for debugging
        policy = self._skill_behavior.actor
        skill = self.wm.heads["object_decoder"](poses=stop_gradient(self._target_pos))["prior"][0][0]
        inp = torch.cat([feat, skill], dim=-1)

        actor = policy(inp)
        if eval_mode:
            action = actor.mean
        else:
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state
