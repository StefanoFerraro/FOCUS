import torch.nn as nn
import torch

from agent.utils import * 
import numpy as np

from agent.focus import FocusAgent
from agent.skill_utils import SkillActorCritic
from agent.dreamer import ConditionedAgent

Module = nn.Module

class FocusPCPAgent(FocusAgent, ConditionedAgent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)

        self._reward_fn = cfg.agent.reward_fn
        self._skill_strategy = "task" # place holder does not have any affect on the agent
        self._target_skill = self._target_pos[0][0]
        self.skill_dim = [self._target_skill.shape[-1]]

        self._skill_behavior = SkillActorCritic(cfg, self.act_spec, self.tfstep, skill_dim=self.skill_dim, 
                                                    sampling_strategy=self._skill_strategy, solved_meta={'skill' : self._target_skill}) 

        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
    
    def update(self, data, step, which_policy='expl'):
        
        if which_policy not in ['expl', 'task', 'both']:
            raise ValueError(f"which_policy must be one of ['expl', 'task', 'both'], got {which_policy}")
        
        dist =torch.exp(- torch.linalg.norm(data["objects_pos"] - self._target_pos, axis=-1) / torch.linalg.norm(self._target_pos))
        data["reward"] = dist # substitute reward signal from the environment with the distance from the asigned target (conditioned on) and the current position
        state, outputs, metrics = self.update_wm(data, step)
                
        start = outputs["post"]
        start = {k: stop_gradient(v) for k, v in start.items()}
        
        self._target_skill = self._target_pos
        self._skill_behavior.solved_meta['skill'] = self._target_skill
        reward_fn = getattr(self, f'{self._reward_fn}_reward_fn')
        
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
                    self.wm, start, data["is_terminal"], reward_fn)
                )
        else:
            metrics.update(
                self._skill_behavior.update(
                    self.wm, start, data["is_terminal"], reward_fn)
                )

            metrics.update(
                    self._expl_behavior.update(
                        self.wm, start, data["is_terminal"], self.expl_reward_fn
                    )
                )
        
        return state, metrics