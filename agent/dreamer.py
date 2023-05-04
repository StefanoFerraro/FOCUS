import torch.nn as nn
import torch
from torch.nn.functional import normalize

import utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np


def stop_gradient(x):
    return x.detach()


Module = nn.Module


class DreamerAgent(Module):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.env = self.cfg["task"][:2]
        self.cfg.update(**kwargs)
        self.obs_space = obs_space
        self.act_spec = act_spec
        self.tfstep = None
        self._use_amp = cfg.precision == 16
        self.device = cfg.device
        self.act_dim = act_spec.shape[0]

        self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep)
        self._task_behavior = ActorCritic(cfg, self.act_spec, self.tfstep)
        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
        self.reward_coeff = cfg.reward_coeff

        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(
            rms,
            cfg.apt.knn_clip,
            cfg.apt.knn_k,
            cfg.apt.knn_avg,
            cfg.apt.knn_rms,
            self.device,
        )

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
        if eval_mode:
            actor = self._task_behavior.actor(feat)
            action = actor.mean
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def update_wm(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        outputs["is_terminal"] = data["is_terminal"]
        metrics.update(mets)
        return state, outputs, metrics

    def update(self, data, step):
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        # Don't train the policy/value if just using MPC
        if getattr(self.cfg, "mpc", False) and (
            not self.cfg.mpc_opt.use_value
        ):
            return state, metrics
        start = {k: stop_gradient(v) for k, v in start.items()}

        # reward_fn = lambda seq: self.wm.heads["reward"](
        # seq["feat"]
        # ).mean  # .mode()

        # reward_fn = lambda seq: (
        #     torch.max(
        #         torch.abs(
        #             self.wm.heads["object_decoder"](
        #                 seq["feat"], only_mlp=True
        #             )["objects_pos"][0].mean
        #         )
        #         - torch.tensor([0, 0, 0.8], device=seq["feat"].device),
        #         2,
        #     ).values.unsqueeze(-1)
        # )  # displace cubeA

        # reward_fn = (
        #     lambda seq: self.wm.heads["object_decoder"](
        #         seq["feat"], only_mlp=True
        #     )["objects_pos"][0]
        #     .mean[:, :, 1]
        #     .unsqueeze(-1)
        # )  # move cubeA right (positive position)

        metrics.update(
            self._task_behavior.update(
                self.wm, start, data["is_terminal"], self.reward_fn
            )
        )
        return state, metrics

    def compute_intr_reward(self, seq):
        rep = stop_gradient(seq)
        B, T, _ = rep.shape
        rep = rep.reshape(B * T, -1)
        reward = self.pbe(rep, cdist=True)
        reward = reward.reshape(B, T, 1)
        return reward

    def reward_fn(self, seq):

        rw_dict = {"rw_mov", "rw_dist_obj", "rw_intr"}

        obj_id = 0
        obj_poses = self.wm.heads["object_decoder"](
            seq["feat"], only_mlp=True
        )["objects_pos"].mean

        obj_pos = obj_poses[:, :, obj_id]

        # "robot0_eef_pos" x, y, z is located at index 21 in full proprio_state
        id_eef = 21 if self.env == "rs" else 18
        gripper_pos = self.wm.heads["decoder"](seq["feat"])["proprio"].mean[
            :, :, id_eef : id_eef + 3
        ]

        rw_mov = obj_pos[:, :, 1].unsqueeze(
            -1
        )  # reward for moving cubeA right (positive position)

        rw_dist_obj = torch.sum(
            ((gripper_pos - obj_pos) ** 2), dim=2
        ).unsqueeze(-1)

        instances = obj_poses.shape[2]
        obj_onehot = torch.eye(
            instances + 1, device=seq["feat"].device
        ).repeat(*seq["feat"].shape[:2], 1, 1)

        x, _ = self.wm.heads["object_decoder"].object_latent_extractor(
            seq["feat"], obj_onehot
        )

        rw_intr = self.compute_intr_reward(x[:, :, obj_id])

        rw = (
            self.reward_coeff["rw_mov"] * rw_mov
            - self.reward_coeff["rw_dist_obj"] * rw_dist_obj
            + self.reward_coeff["rw_intr"] * rw_intr
        )

        rw_dict = {
            "rw_mov": rw_mov,
            "rw_dist_obj": rw_dist_obj,
            "rw_intr": rw_intr,
        }

        return rw, rw_dict

    def report(self, data):
        report = {}
        text = {}
        data = self.wm.preprocess(data)
        with torch.no_grad():
            for key in self.wm.heads["decoder"].cnn_keys:
                name = key.replace("/", "_")
                report[f"{name}"] = self.wm.video_pred(data, key, "decoder")

            # for key in self.wm.heads["decoder"].mlp_keys:
            #     name = key.replace("/", "_")
            #     text[f"{name}"] = self.wm.proprio_pred(
            #         data, key, "decoder", nvid=1
            #     )

            for key in self.wm.heads["object_decoder"].cnn_keys:
                name = key.replace("/", "_")
                report[f"{name}"] = self.wm.video_pred(
                    data, key, "object_decoder", nvid=2
                )

            # for key in self.wm.heads["object_decoder"].mlp_keys:
            #     name = key.replace("/", "_")
            #     text[f"{name}"] = self.wm.object_pos(
            #         data, key, "object_decoder", nvid=1
            #     )

        return report, text

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

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

        if init_critic:
            print(f"Copying the pretrained critic")
            utils.hard_update_params(
                other._task_behavior.critic, self._task_behavior.critic
            )
            if self.cfg.slow_target:
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
            state = self.wm.rssm.img_step(
                {k: v[-1] for k, v in seq.items()}, action
            )
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
        disc = self.cfg.discount * torch.ones(
            list(seq["feat"].shape[:-1]) + [1], device=self.device
        )

        lambda_ret = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
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
        num_pi_trajs = int(
            self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples
        )
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
        mean = torch.zeros(
            self.cfg.mpc_opt.horizon, self.act_dim, device=self.device
        )
        std = 2 * torch.ones(
            self.cfg.mpc_opt.horizon, self.act_dim, device=self.device
        )
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
            value = self.estimate_value(
                start, actions, self.cfg.mpc_opt.horizon
            )
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0
            ).indices
            elite_value, elite_actions = (
                value[elite_idxs],
                actions[:, elite_idxs],
            )

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(
                self.cfg.mpc_opt.temperature * (elite_value - max_value)
            )
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0)
                    * (elite_actions - _mean.unsqueeze(1)) ** 2,
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
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]
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
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.cfg = config
        self.device = config.device
        self.tfstep = tfstep
        self.encoder = common.Encoder(shapes, **config.encoder)

        # Computing embed dim
        with torch.no_grad():
            zeros = {k: torch.zeros((1,) + v) for k, v in shapes.items()}
            outs = self.encoder(zeros)
            embed_dim = outs.shape[1]
        self.embed_dim = embed_dim
        self.rssm = common.EnsembleRSSM(
            **config.rssm,
            action_dim=act_dim,
            embed_dim=embed_dim,
            device=self.device,
        )
        self.heads = {}
        self._use_amp = config.precision == 16
        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch
        self.inp_size = inp_size
        self.heads["decoder"] = common.Decoder(
            shapes, **config.decoder, embed_dim=inp_size
        )
        self.heads["object_decoder"] = common.ObjDecoder(
            shapes, **config.object_decoder, embed_dim=inp_size
        )

        self.heads["reward"] = common.MLP(inp_size, (1,), **config.reward_head)
        if config.pred_discount:
            self.heads["discount"] = common.MLP(
                inp_size, (1,), **config.discount_head
            )
        for name in config.grad_heads:
            assert name in self.heads, name
        self.grad_heads = config.grad_heads
        self.heads = nn.ModuleDict(self.heads)
        self.model_opt = common.Optimizer(
            "model",
            self.parameters(),
            **config.model_opt,
            use_amp=self._use_amp,
        )

    def update(self, data, state=None):
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss, self.parameters()))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(
            embed, data["action"], data["is_first"], state
        )
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

            out = (
                head(inp, data["segmentation"])
                if name == "object_decoder"
                else head(inp)
            )

            dists = out if isinstance(out, dict) else {name: out}

            for key, dist in dists.items():
                like = 0

                if key == "segmentation":
                    seg = data[key].permute(0, 1, 3, 4, 2)
                    like = dist.log_prob(seg)

                elif key == "objects_pos":
                    # for el in range(data[key].shape[2]):
                    like = dist.log_prob(data[key])
                elif key == "rgb" or key == "depth":
                    masks = data["segmentation"]
                    chs = data[key].shape[2]
                    for i in range(masks.shape[2]):
                        if i == 0:
                            images = (
                                masks[:, :, i]
                                .unsqueeze(2)
                                .repeat(1, 1, chs, 1, 1)
                            ) * data[key]
                        else:
                            m = (
                                masks[:, :, i]
                                .unsqueeze(2)
                                .repeat(1, 1, chs, 1, 1)
                            ) * data[key]
                            images = torch.cat((images, m), dim=2)
                    like = dist.log_prob(images)
                else:
                    like = dist.log_prob(data[key])

                likes[key] = like
                losses[key] = -like.mean()

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
            state = self.rssm.img_step(
                {k: v[-1] for k, v in seq.items()}, action
            )
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
                true_first *= self.cfg.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.cfg.discount * torch.ones(
                list(seq["feat"].shape[:-1]) + [1], device=self.device
            )
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = torch.cumprod(
            torch.cat(
                [torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0
            ),
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

            if key == "depth":
                value = value / 4.0 - 0.5

            if key == "segmentation":
                value = value * 1.0

            # if key == "objects_pos":
            # value = value * 100.0  # convert units to meters

            # max = torch.tensor([0.1, 0.1, 1.0], device=value.device)
            # min = torch.tensor([-0.1, -0.1, 0.75], device=value.device)

            # value = (value - min) / (max - min)

            obs[key] = value
        obs["reward"] = {
            "identity": nn.Identity(),
            "sign": torch.sign,
            "tanh": torch.tanh,
        }[self.cfg.clip_rewards](obs["reward"])
        obs["discount"] = 1.0 - obs["is_terminal"].float()
        obs["discount"] *= self.cfg.discount
        return obs

    def segmentation_visualization(
        self,
        seg,
        palette=None,
    ):
        gen = torch.Generator(device=seg.device)

        if palette is None:
            # Get random state before set seed,
            # and restore random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = gen.get_state()
            gen.manual_seed(42)

            # random palette
            palette = torch.randint(
                0,
                255,
                (seg.shape[2], 3),  # segmentation channels are in dim 2
                dtype=torch.uint8,
                generator=gen,
                device=seg.device,
            )

            gen.set_state(state)

        seg_perm = seg.permute(0, 1, 3, 4, 2)

        color_seg = torch.zeros(
            (*seg_perm.shape[:4], 3),
            dtype=torch.uint8,
            device=seg.device,
        )

        for label, color in enumerate(palette):
            color_seg[(seg_perm == 1)[..., label]] = color

        return color_seg.permute(0, 1, 4, 2, 3)

    def object_pos(self, data, key, head, nvid=8):

        decoder = self.heads[head]

        truth = data[key][:nvid][0].unsqueeze(1)

        # normalization object_pose
        # max = torch.tensor([0.1, 0.1, 0.9], device=truth.device)
        # min = torch.tensor([-0.1, -0.1, 0.75], device=truth.device)
        # conf = max - min

        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:nvid, 0].unsqueeze(1),
            data["action"][:nvid, 0].unsqueeze(1),
            data["is_first"][:nvid, 0].unsqueeze(1),
        )

        obj_predictions = decoder(self.rssm.get_feat(states))[key]
        objects = self.cfg.objects

        text_out = []
        dict_out = {}
        rows = ["Proprio GT", "Prediction", "Error"]

        for obj in objects:
            dict_out[obj] = {}
            for row in rows:
                dict_out[obj][row] = []

        for i in range(nvid):
            for j, obj_pred in enumerate(obj_predictions):
                GT = (truth[i][0][j]).cpu().numpy()
                pred = (obj_pred.mean[i][0]).cpu().numpy()
                error = pred - GT
                text_out.append(
                    f"Object {objects[j]} GT={GT}, Prediction={pred}, Error={error}"
                )

                dict_out[objects[j]]["Proprio GT"].append(GT)
                dict_out[objects[j]]["Prediction"].append(pred)
                dict_out[objects[j]]["Error"].append(error)

        # return "\t".join(text_out)
        return dict_out

    def proprio_pred(self, data, key, head, nvid=8):

        decoder = self.heads[head]
        truth = data[key][:nvid][0].unsqueeze(1)
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:nvid, 0].unsqueeze(1),
            data["action"][:nvid, 0].unsqueeze(1),
            data["is_first"][:nvid, 0].unsqueeze(1),
        )

        proprio_pred = decoder(self.rssm.get_feat(states))[key]

        text_out = []
        dict_out = {}
        rows = ["Proprio GT", "Prediction", "Error"]

        for row in rows:
            dict_out[row] = []

        for i in range(nvid):
            GT = truth[i][0].cpu().numpy()
            pred = proprio_pred.mean[i][0].cpu().numpy()
            error = pred - GT
            text_out.append(
                f"Proprio GT={GT}, Prediction={pred}, Error={error}"
            )

            dict_out["Proprio GT"].append(GT)
            dict_out["Prediction"].append(pred)
            dict_out["Error"].append(error)

        # return "\t".join(text_out)
        return dict_out

    def video_pred(self, data, key, head, nvid=8):

        if key == "rgb" or key == "depth":
            decoder = self.heads[head]  # B, T, C, H, W
            truth = data[key][:nvid] + 0.5
            embed = self.encoder(data)
            states, _ = self.rssm.observe(
                embed[:nvid, :5],
                data["action"][:nvid, :5],
                data["is_first"][:nvid, :5],
            )

            recon = decoder(
                self.rssm.get_feat(states), data["segmentation"][:nvid, :5]
            )[key].mean[
                :nvid
            ]  # mode

            recon_unmasked = decoder(self.rssm.get_feat(states))[key].mean[
                :nvid
            ]  # mode

            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.rssm.imagine(data["action"][:nvid, 5:], init)

            prior_recon = decoder(
                self.rssm.get_feat(prior), data["segmentation"][:nvid, 5:]
            )[
                key
            ].mean  # mode

            prior_recon_unmasked = decoder(self.rssm.get_feat(prior))[
                key
            ].mean  # mode

            model = torch.clip(
                torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1
            )

            model_unmasked = torch.clip(
                torch.cat(
                    [recon_unmasked[:, :5] + 0.5, prior_recon_unmasked + 0.5],
                    1,
                ),
                0,
                1,
            )

            # create masks for truth
            masks = data["segmentation"]
            chs = truth.shape[2]
            for m in range(masks.shape[2]):
                if m == 0:
                    truth_out = (
                        masks[:nvid, :, m].unsqueeze(2).repeat(1, 1, chs, 1, 1)
                        * truth
                    )
                else:
                    temp = (
                        masks[:nvid, :, m].unsqueeze(2).repeat(1, 1, chs, 1, 1)
                        * truth
                    )
                    truth_out = torch.cat((truth_out, temp), 4)

            # truth = truth_out
            # divide model output
            model = torch.cat(torch.split(model, chs, 2), dim=4)
            model_unmasked = torch.cat(
                torch.split(model_unmasked, chs, 2), dim=4
            )

            video = torch.cat([truth_out, model, model_unmasked], 3)

        elif key == "segmentation":

            decoder = self.heads[head]  # B, T, C, H, W
            truth = data[key][:nvid] + 0.5
            embed = self.encoder(data)
            states, _ = self.rssm.observe(
                embed[:nvid, :5],
                data["action"][:nvid, :5],
                data["is_first"][:nvid, :5],
            )

            recon = decoder(self.rssm.get_feat(states))[key].mean[
                :nvid
            ]  # mode

            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.rssm.imagine(data["action"][:nvid, 5:], init)
            prior_recon = decoder(self.rssm.get_feat(prior))[key].mean  # mode

            model = torch.clip(
                torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1
            )

            model = model.permute(0, 1, 4, 2, 3)

            error = ((model - truth + 1) / 2).mean(axis=2).unsqueeze(dim=2)

            truth_out = self.segmentation_visualization(truth - 0.5)
            model = self.segmentation_visualization(model)

            error = error.repeat(1, 1, truth_out.shape[2], 1, 1)
            video = torch.cat([truth_out, model, error], 3)

        return video


class ActorCritic(Module):
    def __init__(self, config, act_spec, tfstep):
        super().__init__()
        self.cfg = config
        self.act_spec = act_spec
        self.tfstep = tfstep
        self._use_amp = config.precision == 16
        self.device = config.device

        inp_size = config.rssm.deter
        if config.rssm.discrete:
            inp_size += config.rssm.stoch * config.rssm.discrete
        else:
            inp_size += config.rssm.stoch
        self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
        self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
        if self.cfg.slow_target:
            self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer(
            "actor",
            self.actor.parameters(),
            **self.cfg.actor_opt,
            use_amp=self._use_amp,
        )
        self.critic_opt = common.Optimizer(
            "critic",
            self.critic.parameters(),
            **self.cfg.critic_opt,
            use_amp=self._use_amp,
        )
        self.rewnorm = common.StreamNorm(
            **self.cfg.reward_norm, device=self.device
        )

    def update(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.cfg.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = world_model.imagine(self.actor, start, is_terminal, hor)
                reward, reward_dict = reward_fn(seq)

                mets0 = {}  # individual rewards components
                for rw_key, rw in reward_dict.items():
                    _, met = self.rewnorm(rw)
                    met = {f"{rw_key}_{k}": v for k, v in met.items()}
                    mets0.update(met)

                seq["reward"], mets1 = self.rewnorm(reward)
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}
                target, mets2 = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target)
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = {k: stop_gradient(v) for k, v in seq.items()}
                critic_loss, mets4 = self.critic_loss(seq, target)
            metrics.update(
                self.critic_opt(critic_loss, self.critic.parameters())
            )
        metrics.update(**mets0, **mets1, **mets2, **mets3, **mets4)
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
                    1.0
                    if self._updates == 0
                    else float(self.cfg.slow_target_fraction)
                )
                for s, d in zip(
                    self.critic.parameters(), self._target_critic.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
