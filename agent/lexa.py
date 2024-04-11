import torch.nn as nn
import torch

import numpy as np

from agent.utils import MLP
from agent.plan2explore import Plan2Explore
from agent.skill_utils import SkillActorCritic
from functools import singledispatchmethod

def stop_gradient(x):
    return x.detach()

Module = nn.Module
         
class LEXA(Plan2Explore):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)

        self._mode = "train"
        
        # NOTE: Only for debugging
        self._skill_strategy = 'object_context_pose'
        
        self._target_skill = torch.zeros(1,1,self.wm.inp_size)
        self.skill_dim = self.wm.inp_size # the target feature has the same size of feat

        self._skill_behavior = SkillActorCritic(cfg, self.act_spec, self.tfstep, skill_dim=self.skill_dim, 
                                                    sampling_strategy=self._skill_strategy, solved_meta={'skill' : self._target_skill}) 

        self.wm.dd_out_dim = 1
        self.wm.dynamical_distance = MLP(in_shape=self.skill_dim*2, shape=self.wm.dd_out_dim, layers = 4, units = 128, norm="batch1d", dist="none")
        self.wm.dd_cur_idxs, self.wm.dd_goal_idxs = self.get_future_goal_idxs(seq_len = cfg.batch_length, bs = cfg.batch_size)
        self.wm.dd_seq_len = cfg.batch_length
        
        # self._dd_opt = tools.Optimizer('dynamical_distance_opt', config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

        self.wm.get_dynamical_distance_loss = self.get_dynamical_distance_loss
        
        self.wm.model_init() # include the dynamical distance into the model optimizer 
        
        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
    
    def skill_target_extractor(self):
        return self._skill_target_extractor(self._target)
    
    @singledispatchmethod
    def _skill_target_extractor(self, target):
        pass

    @_skill_target_extractor.register
    def _(self, target: torch.Tensor):
        target_skill = self.wm.object_encoder(stop_gradient(target))["prior"]["mean"][:,:,:,:self._shape_skill_latent]
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
                
            f_i = self.wm.rssm.get_feat(latent).unsqueeze(1).unsqueeze(1)
            
        return f_i
    
    def get_future_goal_idxs(self, seq_len, bs):
   
        cur_idx_list = []
        goal_idx_list = []
        
        #generate indices grid
        for cur_idx in range(seq_len):
            for goal_idx in range(cur_idx, seq_len):
                cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
                goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))
            
        return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

    def get_future_goal_idxs_neg_sampling(self, num_negs, seq_len, bs, batch_len):
        cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        for i in range(num_negs):
            goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
        return cur_idxs, goal_idxs

    def get_dynamical_distance_loss(self, _data, corr_factor = None):
        seq_len, bs = _data.shape[:2]
    
        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            # cur_states = torch.expand_dims(torch.gather_nd(_data, cur_idxs),0)
            cur_states = _data[cur_idxs[:,0], cur_idxs[:,1]].unsqueeze(0)
            # goal_states = torch.expand_dims(torch.gather_nd(_data, goal_idxs),0)
            goal_states = _data[goal_idxs[:,0], goal_idxs[:,1]].unsqueeze(0)
            
            pred = self.wm.dynamical_distance(torch.concat([cur_states, goal_states], axis=-1))
        
            _label = torch.tensor(distance, device="cuda:0")/self.wm.dd_seq_len
            loss += torch.mean((_label-pred)**2)

            return loss
        
        #positives
        idxs = np.random.choice(np.arange(len(self.wm.dd_cur_idxs)), self.cfg.agent.dd_num_positives)
        loss = _helper(self.wm.dd_cur_idxs[idxs], self.wm.dd_goal_idxs[idxs], self.wm.dd_goal_idxs[idxs][:,0] - self.wm.dd_cur_idxs[idxs][:,0])

        #negatives
        corr_factor = corr_factor if corr_factor != None else self.cfg.batch_length
        if self.cfg.agent.dd_neg_sampling_factor>0:
            num_negs = int(self.cfg.agent.dd_neg_sampling_factor*self.cfg.agent.dd_num_positives)
        neg_cur_idxs, neg_goal_idxs = self.get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
        loss += _helper(neg_cur_idxs, neg_goal_idxs, torch.ones(num_negs)*seq_len)

        return loss
    
    def object_context_pose_reward_fn(self, seq):
        
        post_obj_state = seq["feat"]        
        
        if self.cfg.agent.distance_mode == "cosine":
            B, T, _, S = self._target_skill.shape
            self._target_skill = self._target_skill.reshape(B*T, S)
            squared_distance = - (torch.einsum("ijl,ijl->ij", (self._target_skill.unsqueeze(0), post_obj_state)) / (torch.norm(post_obj_state, dim=-1) * torch.norm(self._target_skill, dim=-1) + 1e-12))
        elif self.cfg.agent.distance_mode == "temporal":
            I, BT, S = post_obj_state.shape
            self._target_skill = self._target_skill.reshape(BT, S).repeat(I,1,1)
            squared_distance = self.wm.dynamical_distance(torch.concat([post_obj_state, self._target_skill], axis=-1).detach()).squeeze(-1)
        
        met = self.metric_reward_fn(squared_distance, "object_context_pose")
        return - squared_distance.unsqueeze(-1), met
    
    def update(self, data, step, which_policy='expl'):
        
        if which_policy not in ['expl', 'task', 'both']:
            raise ValueError(f"which_policy must be one of ['expl', 'task', 'both'], got {which_policy}")
        
        state, outputs, metrics = self.update_wm(data, step)

        if step >= self.cfg.agent.start_agent_training_after:
            if step == self.cfg.agent.start_agent_training_after:
                print("STARTING AGENT TRAINING!")
                
            start = outputs["post"]
            start = {k: stop_gradient(v) for k, v in start.items()}

            self._target_skill = self.skill_target_extractor()
            if self.cfg.env.batch_sampling:
                self._target_skill = self._target_skill.repeat(1, self.cfg.batch_size, 1, 1)
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

    def act(self, obs, meta, step, eval_mode, state, target_skill=None):
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
            if self.cfg.agent.only_expl_during_training: # split to avoid errors down the line with meta["use_skill_behaviour"]
                policy = self._expl_behavior.actor
                inp = feat
            else:
                if meta["use_skill_behaviour"]:
                    policy = self._skill_behavior.actor 
                    skill = self.skill_target_extractor()
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