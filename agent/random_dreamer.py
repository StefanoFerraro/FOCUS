import torch.nn as nn
import torch

import numpy as np
from copy import deepcopy

from agent.dreamer import DreamerAgent

Module = nn.Module 

class RandomDreamerAgent(DreamerAgent):
  def __init__(self, **kwargs):
      super().__init__(**kwargs)

  def act(self, obs, meta, step, eval_mode, state):
      if eval_mode:
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
            actor = self._task_behavior.actor(feat)
            action = actor.mean
            new_state = (latent, deepcopy(action))
            action = action.cpu().numpy()
      else:
            action = torch.zeros(self.act_spec.shape).uniform_(-1.0, 1.0).numpy(),
            new_state = None 
      return  action[0], new_state

  # def update(self, data, step):
  #   metrics = {}
  #   state, outputs, mets = self.wm.update(data, state=None)
  #   metrics.update(mets)
  #   return state, metrics