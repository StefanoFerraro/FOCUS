import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.dreamer import DreamerAgent, ActorCritic, stop_gradient
import agent.dreamer_utils as common


class Disagreement(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, n_models=5, pred_dim=None
    ):
        super().__init__()
        if pred_dim is None:
            pred_dim = obs_dim
        self.ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, pred_dim),
                )
                for _ in range(n_models)
            ]
        )

    def forward(self, obs, action, next_obs):
        # import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        errors = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            model_error = torch.norm(
                next_obs - next_obs_hat, dim=-1, p=2, keepdim=True
            )
            errors.append(model_error)

        return torch.cat(errors, dim=1)

    def get_disagreement(self, obs, action):
        assert obs.shape[0] == action.shape[0]

        preds = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            preds.append(next_obs_hat)
        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0).mean(dim=-1)


class Plan2Explore(DreamerAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_dim = self.wm.inp_size
        pred_dim = self.wm.embed_dim
        self.hidden_dim = pred_dim
        self.reward_free = True

        self.disagreement = Disagreement(
            in_dim, self.act_dim, self.hidden_dim, pred_dim=pred_dim
        ).to(self.device)

        # optimizers
        self.disagreement_opt = common.Optimizer(
            "disagreement",
            self.disagreement.parameters(),
            **self.cfg.model_opt,
            use_amp=self._use_amp,
        )
        self.disagreement.train()

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

    def update_disagreement(self, obs, action, next_obs, step):
        metrics = dict()

        error = self.disagreement(obs, action, next_obs)

        loss = error.mean()

        metrics.update(
            self.disagreement_opt(loss, self.disagreement.parameters())
        )

        metrics["disagreement_loss"] = loss.item()

        return metrics

    def compute_intr_reward(self, seq):
        obs, action = seq["feat"][:-1], stop_gradient(seq["action"][1:])
        intr_rew = torch.zeros(
            list(seq["action"].shape[:-1]) + [1], device=self.device
        )
        if len(action.shape) > 2:
            B, T, _ = action.shape
            obs = obs.reshape(B * T, -1)
            action = action.reshape(B * T, -1)
            reward = self.disagreement.get_disagreement(obs, action).reshape(
                B, T, 1
            )
        else:
            reward = self.disagreement.get_disagreement(obs, action).unsqueeze(
                -1
            )
        intr_rew[1:] = reward

        mets = {}

        rw_norm, met = self.task_rewnorm(intr_rew)
        met = {f"rw_intr_{k}": v for k, v in met.items()}
        mets.update(met)

        return rw_norm, mets

    def update(self, data, step):
        metrics = {}
        B, T, _ = data["action"].shape
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs["post"]
        start = {k: stop_gradient(v) for k, v in start.items()}
        if self.is_finetune:
            metrics.update(
                self._task_behavior.update(
                    self.wm, start, data["is_terminal"], self.reward_fn
                )
            )
        else:
            T = T - 1
            inp = stop_gradient(outputs["feat"][:, :-1]).reshape(B * T, -1)
            action = data["action"][:, 1:].reshape(B * T, -1)
            out = stop_gradient(outputs["embed"][:, 1:]).reshape(B * T, -1)
            with common.RequiresGrad(self.disagreement):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(
                        self.update_disagreement(inp, action, out, step)
                    )
            metrics.update(
                self._expl_behavior.update(
                    self.wm,
                    start,
                    data["is_terminal"],
                    reward_fn=self.compute_intr_reward,
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
