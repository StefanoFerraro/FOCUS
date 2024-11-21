import utils
from agent.utils import *
from agent.dreamer import DreamerAgent, ActorCritic
from agent.focus import ExplorationAgent
import agent.dreamer_utils as common

class APTDreamerAgent(ExplorationAgent, DreamerAgent):
    def __init__(self, knn_rms, knn_k, knn_avg, knn_clip, **kwargs):
        super().__init__(**kwargs)

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(
            rms, knn_clip, knn_k, knn_avg, knn_rms, self.device
        )
        self._expl_behavior = ActorCritic(self.cfg, self.act_spec, self.tfstep, name='expl')

        self.expl_rewnorm = common.StreamNorm(
            **self.cfg.agent.reward_norm, device=self.device
        )

        self.to(self.cfg.device)
        self.requires_grad_(requires_grad=False)
    
    def expl_reward_fn(self, seq):
        mets = {}
        # Compute reward
        rep = stop_gradient(seq["deter"])
        B, T, _ = rep.shape
        rep = rep.reshape(B * T, -1)
        reward = self.pbe(rep, cdist=True)
        reward = reward.reshape(B, T, 1)

        # Normalize reward
        rw_norm, met = self.expl_rewnorm(reward)

        # Just for renaming the logging
        met = {f"rw_intr_{k}": v for k, v in met.items()}
        mets.update(met)
        return rw_norm, mets

    def update(self, data, step,  which_policy='expl'):
        if which_policy not in ['expl', 'task', 'both']:
            raise ValueError(f"which_policy must be one of ['expl', 'task', 'both'], got {which_policy}")
        
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        # Don't train the policy/value if just using MPC
        if getattr(self.cfg, "mpc", False) and (
            not self.cfg.mpc_opt.use_value
        ):
            return state, metrics
        start = {k: stop_gradient(v) for k, v in start.items()}
        
        reward_fn = getattr(self, self.cfg.agent.reward_fn + "_reward_fn")
        
        if which_policy == 'expl': 
            metrics.update(
                self._expl_behavior.update(
                    self.wm, start, data["is_terminal"], self.expl_reward_fn
                )
            )
        elif which_policy == 'task':
            metrics.update(
                self._task_behavior.update(
                    self.wm, start, data["is_terminal"], reward_fn
                )
            )
        else:
            metrics.update(
                self._expl_behavior.update(
                    self.wm, start, data["is_terminal"], self.expl_reward_fn
                )
            )

            metrics.update(
                self._task_behavior.update(
                    self.wm, start, data["is_terminal"], reward_fn
                )
            )
            
        return state, metrics
