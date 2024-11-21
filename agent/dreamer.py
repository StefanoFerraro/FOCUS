import torch.nn as nn
import torch

import utils
from agent.utils import * 
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np
from scipy import ndimage
from typing import Union
import omegaconf

Module = nn.Module

class Agent(Module):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.env = self.cfg.domain
        self.task = self.cfg.task
        agent_config = {"agent": kwargs}
        self.cfg.update(**agent_config)
        self.obs_space = obs_space
        self.act_spec = act_spec
        
        self.tfstep = None
        self._use_amp = cfg.precision == 16
        self.device = cfg.device
        self.act_dim = act_spec.shape[0]
        
    def update_wm(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        outputs["is_terminal"] = data["is_terminal"]
        metrics.update(mets)
        return state, outputs, metrics
    
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    @staticmethod
    def metric_reward_fn(values, reward_fn_name):
        met = {f"{reward_fn_name}_rw_mean": values.mean(), f"{reward_fn_name}_rw_std": values.std()}
        return met
    
    def task_reward_fn(self, seq):
        rw = self.wm.heads["reward"](seq["feat"]).mean
        met = self.metric_reward_fn(rw, "task")
        return rw, met

class ConditionedAgent(Agent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)
        self._exploration_area=None
        
        self.set_target(cfg.env.object_start_pos)
        
    def update_target(self):
        if not self._exploration_area:
            raise Exception("Set exploration area before target update")
        new_target = np.random.uniform(*self._exploration_area)
        self.set_target(new_target)
        
    def set_target(self, target):
        if isinstance(target, dict):
            self._target_pos = target["objects_pos"]           
            self._target = target
        elif isinstance(target, Union[np.ndarray, list, torch.Tensor, omegaconf.listconfig.ListConfig]):
            target = np.array(target)
            if len(target.shape) == 1: 
                self._target_pos = self._target = torch.tensor([[[target]]], device="cuda", dtype=torch.float)
            elif len(target.shape) == 3:
                self._target_pos = self._target = torch.tensor(target, device="cuda", dtype=torch.float).unsqueeze(-2) # unsqueeze along the object dimension
    
    def get_target_pos(self):
        return self._target_pos
       
    def set_exploration_area(self, exploration_area):
        self._exploration_area = exploration_area
        
    def get_init_exploration_area(self):
        return self.init_exploration_area
    
    def act(self, obs, meta, step, eval_mode, state, **kwargs):
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

        # policy selection based on exploration or finetune mode
        policy = self._skill_behavior.actor
        skill = self._target_pos[0][0]
        inp = torch.cat([feat, skill], dim=-1)

        actor = policy(inp)
        if eval_mode:
            action = actor.mean
        else:
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state
    
        
class DreamerAgent(Agent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)
        
        self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep)
        
        self.wm.model_init()

        self._task_behavior = ActorCritic(cfg, self.act_spec, self.tfstep, name="task")
        
        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
 
    def act(self, obs, meta, step, eval_mode, state, finetune_mode=False):
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
        if eval_mode:
            action = actor.mean
        else:
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state
    
    def dist_reward_fn(self, seq):
        pred = self.wm.heads["decoder"](seq["feat"], only_mlp=True)
        dist = pred["objects_pos"].mean - pred["target"].mean.unsqueeze(-2)
        squared_distance = torch.sum(((dist) ** 2), dim=3)
        met = self.metric_reward_fn(squared_distance, "dist_to_target")
        return - squared_distance, met # want to minimize the distance
    
    def pos_reward_fn(self, seq):
        pos_pred = self.wm.heads["decoder"](seq["feat"], only_mlp=True)["objects_pos"].mean
        # distance from current predicted position to the target
        squared_distance = torch.sum(((pos_pred - self._target_pos) ** 2), dim=3)
        met = self.metric_reward_fn(squared_distance, "pos")
        return - squared_distance.detach(), met # maximization of reward coincide with the minimization of the distance

    def update(self, data, step, **kwargs):
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        # Don't train the policy/value if just using MPC
        if getattr(self.cfg, "mpc", False) and (not self.cfg.mpc_opt.use_value):
            return state, metrics
        
        start = {k: stop_gradient(v) for k, v in start.items()}

        reward_fn = getattr(self, self.cfg.agent.reward_fn + "_reward_fn")
        metrics.update(
            self._task_behavior.update(
                self.wm, start, data["is_terminal"], reward_fn
            )
        )
        return state, metrics

    def report(self, data):
        report = {}
        metrics = {}

        data = self.wm.preprocess(data)
        with torch.no_grad():
            for key in self.wm.heads["decoder"].cnn_keys:
                name = key.replace("/", "_")
                report[f"{name}"] = self.wm.video_pred(data, key, nvid=4)
            
            # Extra debugging metrics  
            # if "objects_pos" in self.wm.heads["decoder"].mlp_keys:
            #     metrics.update(self.wm.prediction_errors(data, "objects_pos", batch_size=10, head_key="decoder"))
            # metrics.update(self.wm.prediction_errors(data, "reward", batch_size=10, head_key="reward"))
            # metrics.update(self.wm.RSSM_errors(data))
        return report, metrics

    def init_from(self, other):
        init_critic = self.cfg.get("init_critic", False)
        init_actor = self.cfg.get("init_actor", True)

        # copy parameters over
        print(f"Copying the pretrained world model")
        utils.hard_update_params(other.wm.rssm, self.wm.rssm)
        utils.hard_update_params(other.wm.encoder, self.wm.encoder)
        utils.hard_update_params(other.wm.heads["decoder"], self.wm.heads["decoder"])

        if init_actor:
            print(f"Copying the pretrained actor")
            utils.hard_update_params(
                other._task_behavior.actor, self._task_behavior.actor
            )

        if init_critic:
            print(f"Copying the pretrained critic")
            utils.hard_update_params(
                other._task_behavior.critic, self._task_behavior.critic
            )
            if self.cfg.agent.slow_target:
                utils.hard_update_params(
                    other._task_behavior._target_critic,
                    self._task_behavior._target_critic,
                )

    @torch.no_grad()
    def estimate_value(self, start, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start["feat"] = self.wm.rssm.get_feat(start)
        start["action"] = torch.zeros_like(actions[0], device=self.device)
        seq = {k: [v] for k, v in start.items()}
        for t in range(horizon):
            action = actions[t]
            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.wm.rssm.get_feat(state)
            for key, value in {
                **state,
                "action": action,
                "feat": feat,
            }.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        reward = self.wm.heads["reward"](seq["feat"]).mean
        if self.cfg.mpc_opt.use_value:
            value = self._task_behavior._target_critic(seq["feat"]).mean
        else:
            value = torch.zeros_like(reward, device=self.device)
        disc = self.cfg.agent.discount * torch.ones(
            list(seq["feat"].shape[:-1]) + [1], device=self.device
        )

        lambda_ret = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.agent.discount_lambda,
            axis=0,
        )

        # First step is lost because the reward is from the start state
        return lambda_ret[1]

    @torch.no_grad()
    def plan(self, obs, meta, step, eval_mode, state, t0=True):
        """
        Plan next action using Dyna-MPC.
        We thank the authors of TD-MPC (https://github.com/nicklashansen/tdmpc), to provide a good reference for implementing our planning strategy.
        """

        # Get Dreamer's state and features
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
        post, prior = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], should_sample
        )
        feat = self.wm.rssm.get_feat(post)

        # Sample policy trajectories
        num_pi_trajs = int(self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples)
        if num_pi_trajs > 0:
            start = {
                k: v.repeat(num_pi_trajs, *list([1] * len(v.shape)))
                for k, v in post.items()
            }
            seq = self.wm.imagine(
                self._task_behavior.actor,
                start,
                None,
                self.cfg.mpc_opt.horizon,
            )
            pi_actions = seq["action"][1:]

        # Initialize state and parameters
        start = {
            k: v.repeat(
                self.cfg.mpc_opt.num_samples + num_pi_trajs,
                *list([1] * len(v.shape)),
            )
            for k, v in post.items()
        }
        mean = torch.zeros(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        std = 2 * torch.ones(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        if not t0 and hasattr(self, "_prev_mean"):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.mpc_opt.iterations):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    self.cfg.mpc_opt.horizon,
                    self.cfg.mpc_opt.num_samples,
                    self.act_dim,
                    device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(start, actions, self.cfg.mpc_opt.horizon)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0
            ).indices
            elite_value, elite_actions = (
                value[elite_idxs],
                actions[:, elite_idxs],
            )

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.mpc_opt.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(self.cfg.mpc_opt.min_std, 2)
            mean, std = (
                self.cfg.mpc_opt.momentum * mean
                + (1 - self.cfg.mpc_opt.momentum) * _mean,
                _std,
            )

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        new_state = (post, a.unsqueeze(0))
        return a.cpu().numpy(), new_state


class WorldModel(Module):
    def __init__(self, config, obs_space, act_dim, tfstep):
        super().__init__()
        self.shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.full_cfg = config
        self.cfg = config.agent.world_model
        self.device = config.device
        self.tfstep = tfstep
        
        self.encoder = common.Encoder(self.shapes, symlog_inputs=self.full_cfg.agent.symlog_inputs, **self.cfg.encoder)
        self.encoder.to(self.device)
        
        # Computing embed dim
        with torch.no_grad():
            zeros = {k: torch.zeros((1,) + v, device=self.device) for k, v in self.shapes.items()}
            
            outs = self.encoder(zeros)
            embed_dim = outs.shape[1]
        self.embed_dim = embed_dim
        self.rssm = common.EnsembleRSSM(
            **self.cfg.rssm,
            action_dim=act_dim,
            embed_dim=embed_dim,
            device=self.device,
        )
        self.heads = {}
        self._use_amp = config.precision == 16
        self.inp_size = self.cfg.rssm.deter
        if self.cfg.rssm.discrete:
            self.inp_size += self.cfg.rssm.stoch * self.cfg.rssm.discrete
        else:
            self.inp_size += self.cfg.rssm.stoch
        self.inp_size = self.inp_size
        self.heads["decoder"] = common.Decoder(
            self.shapes, **self.cfg.decoder, symlog_outputs=self.full_cfg.agent.symlog_inputs, embed_dim=self.inp_size
        )
        self.heads["reward"] = MLP(self.inp_size, (1,), **self.cfg.reward_head)
        if self.cfg.pred_discount:
            self.heads["discount"] = MLP(
                self.inp_size, (1,), **self.cfg.discount_head
            )

    def model_init(self):
        for name in self.cfg.grad_heads:
            assert name in self.heads, name
        self.grad_heads = self.cfg.grad_heads
        self.heads = nn.ModuleDict(self.heads)
        self.model_opt = Optimizer(
            "model",
            self.parameters(),
            **self.cfg.model_opt,
            use_amp=self._use_amp,
        )
        for p in self.heads["reward"]._out.parameters():
            p.data.fill_(0.0)

    def update(self, data, state=None):
        with RequiresGrad(self):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss, self.parameters()))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        data["embed"] = stop_gradient(embed)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
        assert len(kl_loss.shape) == 0 or (
            len(kl_loss.shape) == 1 and kl_loss.shape[0] == 1
        ), kl_loss.shape
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.grad_heads
            inp = feat if grad_head else stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                # if key == "objects_pos":
                    # data[key] = data[key].squeeze(-2)
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()
        
        # in case we are training a lexa model we need to update also the temporal loss
        # if self.cfg.name == "lexa" and self.full_cfg.agent.distance_mode == "temporal": 
        #     states, _ = self.rssm.observe(
        #         self.encoder(data),
        #         data["action"],
        #         data["is_first"],
        #     )
        #     feat = self.rssm.get_feat(states)
        #     losses["dynamical_distance"] = self.get_dynamical_distance_loss(feat, corr_factor=1)
            
        model_loss = sum(
            self.cfg.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed,
            feat=feat,
            post=post,
            prior=prior,
            likes=likes,
            kl=kl_value,
        )
        metrics = {f"{name}_loss": value for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics
    
    def imagine(
        self,
        policy,
        start,
        is_terminal,
        horizon,
        task_cond=None,
        eval_policy=False,
    ):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start["feat"] = self.rssm.get_feat(start)
        inp = (
            start["feat"]
            if task_cond is None
            else torch.cat([start["feat"], task_cond], dim=-1)
        )
        start["action"] = torch.zeros_like(
            policy(inp).mean, device=self.device
        )  # .mode())
        seq = {k: [v] for k, v in start.items()}
        if task_cond is not None:
            seq["task"] = [task_cond]
        for _ in range(horizon):
            inp = (
                seq["feat"][-1]
                if task_cond is None
                else torch.cat([seq["feat"][-1], task_cond], dim=-1)
            )
            action = (
                policy(stop_gradient(inp)).sample()
                if not eval_policy
                else policy(stop_gradient(inp)).mean
            )
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {
                **state,
                "action": action,
                "feat": feat,
            }.items():
                seq[key].append(value)
            if task_cond is not None:
                seq["task"].append(task_cond)
        # shape will be (T, B, *DIMS)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal)
                true_first *= self.full_cfg.agent.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.full_cfg.agent.discount * torch.ones(
                list(seq["feat"].shape[:-1]) + [1], device=self.device
            )
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0),
            0,
        )
        return seq

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype in [np.uint8, torch.uint8] and "rgb" in key:
                value = value / 255.0 - 0.5

            # normalization and scalling of depth based on observed max value over entire set
            if key == "depth":
                value = (
                    value / value.max() - 0.5
                )  # ASSUME that camera is static and that max depth is fixed

            if key == "segmentation":
                value = value * 1.0

            obs[key] = value
        obs["reward"] = {
            "identity": nn.Identity(),
            "sign": torch.sign,
            "tanh": torch.tanh,
        }[self.full_cfg.agent.clip_rewards](obs["reward"])
        obs["discount"] = 1.0 - obs["is_terminal"].float()
        obs["discount"] *= self.full_cfg.agent.discount
        return obs

    def RSSM_errors(self, data):
        metrics = {"prior_error": [], "prior_recon_error": [], "prior_recon_masked_error": []}
        error_steps = [1, 3, 5, 8, 10]
        decoder = self.heads["decoder"]  # B, T, C, H, W
        truth = data["rgb"] + 0.5
        embed = self.encoder(data)
        imag_start_step = 5
        over_steps = lambda x : [x[:s] for s in error_steps]
        
        post, prior = self.rssm.observe(
            embed,
            data["action"],
            data["is_first"],
        )
    
        recon = decoder(self.rssm.get_feat(post))["rgb"].mean
        
        init = {k: v[:, imag_start_step] for k, v in post.items()}
        for s in error_steps:
            # mse errors in the predictions
            pred_prior = self.rssm.imagine(data["action"][:, imag_start_step:imag_start_step+s], init)
            prior_recon = decoder(self.rssm.get_feat(pred_prior))["rgb"].mean
            
            prior_error = torch.mean((self.rssm.get_feat(pred_prior)[:, -1] - self.rssm.get_feat(prior)[:, imag_start_step + s]) ** 2)
            metrics["prior_error"].append(prior_error.item())
            prior_recon_error = torch.mean((prior_recon[:, -1] - truth[:, imag_start_step + s]) ** 2)
            metrics["prior_recon_error"].append(prior_recon_error.item())
            
            # mse objects errors in the predictions
            # Mask of object -> dilate mask -> compute mse over selected region
            if "segmentation" in data.keys():
                seg = data["segmentation"][:,:,0]
                struct = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]) # structure to dilatate only on the last two channels, 2D propagation, batches are keeped separate
                dilated_seg = ndimage.binary_dilation(seg.detach().cpu().numpy(), struct, iterations=3)
                dilated_seg_rgb_axis = np.repeat(dilated_seg[:, imag_start_step + s, np.newaxis], 3, axis=1)
                
                truth_masked = truth[:, imag_start_step + s].detach().cpu().numpy() * dilated_seg_rgb_axis
                prior_recon_masked = prior_recon[:, -1].detach().cpu().numpy() * dilated_seg_rgb_axis
                
                prior_recon_masked_error = np.mean((truth_masked - prior_recon_masked) ** 2)
                metrics["prior_recon_masked_error"].append(prior_recon_masked_error)

        return metrics
    
    def video_pred(self, data, key, nvid=8):
        decoder = self.heads["decoder"]  # B, T, C, H, W
        truth = data[key][:nvid] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:nvid, :5],
            data["action"][:nvid, :5],
            data["is_first"][:nvid, :5],
        )
        recon = decoder(self.rssm.get_feat(states))[key].mean[:nvid]  # mode
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:nvid, 5:], init)
        prior_recon = decoder(self.rssm.get_feat(prior))[key].mean  # mode
        model = torch.clip(torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1)
        error = (model - truth + 1) / 2
        
        if getattr(self, "recon_skills", False):
            prior_feat = self.rssm.get_feat(prior)
            if self.skill_module.discrete_skills:
                B, T, _ = prior["deter"].shape
                z_e = self.skill_module.skill_encoder(
                    prior["deter"].reshape(B * T, -1)
                ).mean
                z_q, _ = self.skill_module.emb(z_e, weight_sg=True)
                latent_skills = z_q.reshape(B, T, -1)
            else:
                latent_skills = self.skill_module.skill_encoder(prior["deter"]).mean
                latent_skills = latent_skills / torch.norm(
                    latent_skills, dim=-1, keepdim=True
                )

            x = deter = self.skill_module.skill_decoder(latent_skills).mean

            stats = self.rssm._suff_stats_ensemble(x)
            index = torch.randint(0, self.rssm._ensemble, ())
            stats = {k: v[index] for k, v in stats.items()}
            dist = self.rssm.get_dist(stats)
            stoch = dist.sample()
            prior = {"stoch": stoch, "deter": deter, **stats}
            skill_recon = decoder(self.rssm.get_feat(prior))[key].mean  # mode
            error = torch.clip(
                torch.cat([recon[:, :5] + 0.5, skill_recon + 0.5], 1), 0, 1
            )

        video = torch.cat([truth, model, error], 3)
        B, T, C, H, W = video.shape
        return video
    
    def prediction_errors(self, data, key, batch_size=10, head_key="decoder"):
        decoder = self.heads[head_key]  # B, T, C, H, W
        truth = data[key][:batch_size]
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:batch_size],
            data["action"][:batch_size],
            data["is_first"][:batch_size],
        )
        
        feat = self.rssm.get_feat(states)
        if head_key == "decoder":
            pred = decoder(feat[:batch_size], only_mlp=True)["objects_pos"].mean
        else:
            pred = decoder(feat[:batch_size]).mean            
        
        MSE = torch.mean((pred - truth) ** 2)
        MAE = torch.mean(torch.abs(pred - truth))
        NMAE = torch.mean(torch.abs(pred - truth) / (torch.mean(torch.abs(truth)) + 1e-12))
        exp_dist = torch.exp(-torch.linalg.norm(pred - truth))
        exp_dist_norm = torch.exp(-torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))
        
        output = {"MSE": MSE, "MAE": MAE, "NMAE": NMAE, "exp_dist": exp_dist, "exp_dist_norm": exp_dist_norm}
        
        return {f"{key}_{k}_error": v for k, v in output.items()}
        
class ActorCritic(Module):
    def __init__(self, config, act_spec, tfstep, name="default"):
        super().__init__()
        self.cfg = config.agent
        self.act_spec = act_spec
        self.tfstep = tfstep
        self._use_amp = config.precision == 16
        self.hor = config.imag_horizon
        self.device = config.device

        inp_size = self.cfg.world_model.rssm.deter
        if self.cfg.world_model.rssm.discrete:
            inp_size += self.cfg.world_model.rssm.stoch * self.cfg.world_model.rssm.discrete
        else:
            inp_size += self.cfg.world_model.rssm.stoch
        self.actor = MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
        self.critic = MLP(inp_size, (1,), **self.cfg.critic)
        if self.cfg.slow_target:
            self._target_critic = MLP(inp_size, (1,), **self.cfg.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = Optimizer(
            f"{name}_actor",
            self.actor.parameters(),
            **self.cfg.actor_opt,
            use_amp=self._use_amp,
        )
        self.critic_opt = Optimizer(
            f"{name}_critic",
            self.critic.parameters(),
            **self.cfg.critic_opt,
            use_amp=self._use_amp,
        )
        self.rewnorm = StreamNorm(**self.cfg.reward_norm, device=self.device)
        for p in self.critic._out.parameters():
            p.data.fill_(0.0)
        for p in self._target_critic._out.parameters():
            p.data.fill_(0.0)

    def update(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = world_model.imagine(self.actor, start, is_terminal, self.hor)
                
                # seq["reward"], mets1 = reward_fn(seq) # TODO: check if norm is necessary
                reward, _ = reward_fn(seq)
                seq['reward'], mets1 = self.rewnorm(reward)
                mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        
                target, mets2 = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target)
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        with RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = {k: stop_gradient(v) for k, v in seq.items()}
                critic_loss, mets4 = self.critic_loss(seq, target)
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):  # , step):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(stop_gradient(seq["feat"][:-2]))
        if self.cfg.actor_grad == "dynamics":
            objective = target[1:]
        elif self.cfg.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mean  # .mode()
            advantage = stop_gradient(target[1:] - baseline)
            objective = (
                policy.log_prob(stop_gradient(seq["action"][1:-1]))[:, :, None]
                * advantage
            )
        elif self.cfg.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mean  # .mode()
            advantage = stop_gradient(target[1:] - baseline)
            objective = (
                policy.log_prob(stop_gradient(seq["action"][1:-1]))[:, :, None]
                * advantage
            )
            mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.cfg.actor_grad)
        ent = policy.entropy()[:, :, None]
        ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = stop_gradient(seq["weight"])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_ent"] = ent.mean()
        metrics["actor_ent_scale"] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1])
        target = stop_gradient(target)
        weight = stop_gradient(seq["weight"])
        critic_loss = -(dist.log_prob(target)[:, :, None] * weight[:-1]).mean()
        metrics = {"critic": dist.mean.mean()}  # .mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = seq["reward"]
        disc = seq["discount"]
        value = self._target_critic(seq["feat"]).mean  # .mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0,
        )
        metrics = {}
        metrics["critic_slow"] = value.mean()
        metrics["critic_target"] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.cfg.slow_target:
            if self._updates % self.cfg.slow_target_update == 0:
                mix = (
                    1.0 if self._updates == 0 else float(self.cfg.slow_target_fraction)
                )
                for s, d in zip(
                    self.critic.parameters(), self._target_critic.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
