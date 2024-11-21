import torch.nn as nn
import torch

from agent.utils import * 
import numpy as np

from agent.focus import FocusAgent
from agent.skill_utils import SkillActorCritic
from agent.dreamer import ConditionedAgent

from functools import singledispatchmethod

Module = nn.Module
         
class FocusLCPAgent(FocusAgent, ConditionedAgent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)

        self._skill_strategy = 'object_context_pose'
        self._shape_skill_latent = self.wm._shape_skill_latent
        self._target_skill = self.skill_target_extractor()
        self.skill_dim = [self._target_skill.shape[-1]]

        self._skill_behavior = SkillActorCritic(cfg, self.act_spec, self.tfstep, skill_dim=self.skill_dim, 
                                                    sampling_strategy=self._skill_strategy, solved_meta={'skill' : self._target_skill}) 

        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
    
    def skill_target_extractor(self):
        return self._skill_target_extractor(self._target)
    
    @singledispatchmethod
    def _skill_target_extractor(self, target):
        pass

    @_skill_target_extractor.register
    def _(self, target: torch.Tensor):
        target_skill = self.wm.object_encoder(stop_gradient(target))["prior"]["mean"][:,:,:, :self._shape_skill_latent]
        return target_skill
    
    @_skill_target_extractor.register
    def _(self, target: dict):
        tensor_target_obs =  target

        with torch.no_grad():
            embed = self.wm.encoder(self.wm.preprocess(tensor_target_obs))
            
            warmup_cycles = 1
            latent = self.wm.rssm.initial(len(target["reward"]))
            
            for _ in range(warmup_cycles):
                latent, _ = self.wm.rssm.obs_step(
                    latent,
                    tensor_target_obs["action"][:,0],
                    embed[:,0],
                    tensor_target_obs["is_first"][:,0],
                    should_sample=True
                    )
                
            f_i = self.wm.rssm.get_feat(latent).unsqueeze(1)
            target_skill = self.wm.heads["object_decoder"].object_latent_extractor(f_i.detach())["post"]["mean"][:,:,0,:self._shape_skill_latent].unsqueeze(-2)
            
        return target_skill
        
    def object_context_position_reward_fn(self, seq):
        obj_id, obj_goal_pos = torch.split(seq['skill'] , self.skill_dim, dim=-1) 
        
        obj_poses = self.wm.heads["object_decoder"](seq["feat"], only_mlp=True)[
            "objects_pos"
        ].mean
        T, B, O, P = obj_poses.shape
        obj_pos = obj_poses.reshape(T*B, O, P)[torch.arange(T*B), torch.argmax(obj_id,-1).reshape(T*B)].reshape(T,B,P)
        squared_distance = torch.sum(((obj_goal_pos - obj_pos) ** 2), dim=2).unsqueeze(-1) 
        met = self.metric_reward_fn(squared_distance, "object_context_position")
        return -squared_distance, met

    def distance_to_object_reward_fn(self, seq):
        obj_id = 0
        obj_poses = self.wm.heads["object_decoder"](seq["feat"])["objects_pos"].mean
        obj_pos = obj_poses[:, :, obj_id]
        
        # "robot0_eef_pos" x, y, z is located at index 21 in full proprio_state
        id_eef = 21 if self.env == "rs" else 18
        gripper_pos = self.wm.heads["decoder"](seq["feat"])["proprio"].mean[
                :, :, id_eef : id_eef + 3
            ]
        
        rw_dist_obj = torch.exp(-torch.linalg.norm(gripper_pos - obj_pos, dim=2)).unsqueeze(-1)
        met = self.metric_reward_fn(rw_dist_obj, "distance_to_object")
        return rw_dist_obj, met
        
    def object_context_pose_reward_fn(self, seq):
        feat = seq["stoch"].flatten(-2) if self.stoch_only else seq["feat"]
        
        post_obj_state = self.wm.heads["object_decoder"].object_latent_extractor(stop_gradient(feat))["post"]["mean"][:,:,0,:self._shape_skill_latent] #consider only first object
        if self.cfg.agent.distance_mode == "mse":
            squared_distance = torch.sum(((post_obj_state - self._target_skill) ** 2), dim=2)
        elif self.cfg.agent.distance_mode == "cosine":
            B, T, _, S = self._target_skill.shape
            self._target_skill = self._target_skill.reshape(B*T, S)
            squared_distance = - (torch.einsum("ijl,ijl->ij", (self._target_skill.unsqueeze(0), post_obj_state)) / (torch.norm(post_obj_state, dim=-1) * torch.norm(self._target_skill, dim=-1) + 1e-12))
        met = self.metric_reward_fn(squared_distance, "object_context_pose")
        return - squared_distance.unsqueeze(-1), met
    
    def update(self, data, step, which_policy='expl'):
        
        if which_policy not in ['expl', 'task', 'both']:
            raise ValueError(f"which_policy must be one of ['expl', 'task', 'both'], got {which_policy}")
        
        state, outputs, metrics = self.update_wm(data, step)
                
        start = outputs["post"]
        start = {k: stop_gradient(v) for k, v in start.items()}
        
        self._target_skill = self.skill_target_extractor()
        if self.cfg.batch_sampling:
            self._target_skill = self._target_skill.repeat(1, self.cfg.batch_length, 1, 1)
    
        self._skill_behavior.solved_meta['skill'] = self._target_skill
        
        # agent update based on the achievement on the given skill 
        if which_policy == 'expl':
            metrics.update(
                self._expl_behavior.update(
                    self.wm, start, data["is_terminal"], self.expl_reward_fn
                )
            )
        elif which_policy == 'task':
            metrics.update(
                self._skill_behavior.update(
                    self.wm, start, data["is_terminal"], getattr(self, f'{self._skill_strategy}_reward_fn')
                )
            )
        else:
            metrics.update(
                self._skill_behavior.update(
                    self.wm, start, data["is_terminal"], getattr(self, f'{self._skill_strategy}_reward_fn')
                )
            )
            metrics.update(
                    self._expl_behavior.update(
                        self.wm, start, data["is_terminal"], self.expl_reward_fn
                    )
                )
        
        return state, metrics

    def act(self, obs, meta, step, eval_mode, state, target_skill=None, **kwargs):
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

        if eval_mode:
            policy =  self._skill_behavior.actor
            if target_skill is None:
                skill = self.skill_target_extractor()[0,0]
            else:
                skill = target_skill
            
            inp = torch.cat([feat, skill], dim=-1)
        else:
            policy = self._expl_behavior.actor
            inp = feat

        actor = policy(inp)
        if eval_mode:
            action = actor.mean
        else:
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state