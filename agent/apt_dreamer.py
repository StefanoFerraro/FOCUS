import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.dreamer import DreamerAgent, ActorCritic, stop_gradient
import agent.dreamer_utils as common


class APTDreamerAgent(DreamerAgent):
    def __init__(self, knn_rms, knn_k, knn_avg, knn_clip, **kwargs):
        super().__init__(**kwargs)
        self.reward_free = True

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(
            rms, knn_clip, knn_k, knn_avg, knn_rms, self.device
        )
        self._expl_behavior = ActorCritic(self.cfg, self.act_spec, self.tfstep)

        self.expl_rewnorm = common.StreamNorm(
            **self.cfg.reward_norm, device=self.device
        )
        self.is_finetune = kwargs["is_finetune"]

        self.to(self.cfg.device)
        self.requires_grad_(requires_grad=False)

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

        policy = (
            self._task_behavior.actor
            if self.is_finetune
            else self._expl_behavior.actor
        )

        if eval_mode:
            actor = policy(feat)
            action = actor.mean
        else:
            actor = policy(feat)
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def expl_reward_fn(self, seq):
        rep = stop_gradient(seq["deter"])
        B, T, _ = rep.shape
        rep = rep.reshape(B * T, -1)
        reward = self.pbe(rep, cdist=True)
        reward = reward.reshape(B, T, 1)

        mets = {}

        rw_norm, met = self.task_rewnorm(reward)
        met = {f"rw_intr_{k}": v for k, v in met.items()}
        mets.update(met)

        return rw_norm, mets

    def update(self, data, step):
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        # Don't train the policy/value if just using MPC
        if getattr(self.cfg, "mpc", False) and (
            not self.cfg.mpc_opt.use_value
        ):
            return state, metrics
        start = {k: stop_gradient(v) for k, v in start.items()}

        if self.is_finetune:
            metrics.update(
                self._task_behavior.update(
                    self.wm, start, data["is_terminal"], self.reward_fn
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
                    self.wm, start, data["is_terminal"], self.reward_fn
                )
            )
        return state, metrics

    def init_from(self, other):
        init_critic = self.cfg.get("init_critic", False)
        init_actor = self.cfg.get("init_actor", True)

        # copy parameters over
        print(f"Copying the pretrained world model")
        utils.hard_update_params(other.wm.rssm, self.wm.rssm)
        utils.hard_update_params(other.wm.encoder, self.wm.encoder)
        utils.hard_update_params(
            other.wm.heads["decoder"], self.wm.heads["decoder"]
        )

        if init_actor:
            print(f"Copying the pretrained actor")
            utils.hard_update_params(
                other._task_behavior.actor, self._task_behavior.actor
            )
            utils.hard_update_params(
                other._expl_behavior.actor, self._expl_behavior.actor
            )

        if init_critic:
            print(f"Copying the pretrained critic")
            utils.hard_update_params(
                other._task_behavior.critic, self._task_behavior.critic
            )
            utils.hard_update_params(
                other._expl_behavior.critic, self._expl_behavior.critic
            )
            if self.cfg.slow_target:
                utils.hard_update_params(
                    other._task_behavior._target_critic,
                    self._task_behavior._target_critic,
                )
                utils.hard_update_params(
                    other._expl_behavior._target_critic,
                    self._expl_behavior._target_critic,
                )
