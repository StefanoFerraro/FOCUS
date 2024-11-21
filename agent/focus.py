import torch.nn as nn
import torch

import utils
from agent.utils import * 
import agent.dreamer_utils as common
import numpy as np

from agent.dreamer import Agent, ActorCritic, WorldModel

Module = nn.Module

class ExplorationAgent(Agent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)

        self.obj_instances = self.obs_space["objects_pos"].shape[0]
        self.stoch_only = not self.cfg.agent.world_model.rssm.full_posterior

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

        # policy selection based on eval mode or finetune mode
        policy = policy_selector(self._task_behavior.actor, self._expl_behavior.actor, [eval_mode, finetune_mode]) 

        actor = policy(feat)
        if eval_mode:
            action = actor.mean
        else:
            action = actor.sample()

        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

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
                
class FocusAgent(ExplorationAgent):
    def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
        super().__init__(name, cfg, obs_space, act_spec, **kwargs)

        self.obj_instances = self.obs_space["objects_pos"].shape[0]
        self.stoch_only = not self.cfg.agent.world_model.rssm.full_posterior

        self.tfstep = None
        self._use_amp = cfg.precision == 16
        self.device = cfg.device
        self.act_dim = act_spec.shape[0]

        self.wm = OCWorldModel(cfg, obs_space, self.act_dim, self.tfstep)
        self.wm.model_init()

        self._expl_behavior = ActorCritic(cfg, self.act_spec, self.tfstep, name="expl")
        self._task_behavior = ActorCritic(cfg, self.act_spec, self.tfstep, name="task")

        self.reward_coeff = cfg.agent.reward_coeff
        self.rw_dict = {"rw_task", "rw_intr"}

        self.rewnorm_dict = {}
        for k in self.rw_dict:
            self.rewnorm_dict[k] = common.StreamNorm(
                **cfg.agent.reward_norm, device=self.device
            )

        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(
            rms,
            cfg.agent.apt.knn_clip,
            cfg.agent.apt.knn_k,
            cfg.agent.apt.knn_avg,
            cfg.agent.apt.knn_rms,
            self.device,
        )
        
        self.to(cfg.device)
        self.requires_grad_(requires_grad=False)
    
    def update(self, data, step, which_policy='expl'):
        if which_policy not in ['expl', 'task', 'both']:
            raise ValueError(f"which_policy must be one of ['expl', 'task', 'both'], got {which_policy}")
        
        state, outputs, metrics = self.update_wm(data, step)

        start = outputs["post"]
        start = {k: stop_gradient(v) for k, v in start.items()}

        reward_fn = getattr(self, self.cfg.agent.reward_fn + "_reward_fn")
        
        # update based on mode, save compute time
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

    def compute_intr_reward(self, seq):
        rep = stop_gradient(seq)
        B, T, _ = rep.shape
        rep = rep.reshape(B * T, -1)
        reward = self.pbe(rep, cdist=True)
        reward = reward.reshape(B, T, 1)
        return reward
    
    def pos_reward_fn(self, seq):
        pos_pred = self.wm.heads["object_decoder"](seq["feat"], only_mlp=True)["objects_pos"].mean
        # distance from current predicted position to the target
        squared_distance = torch.sum(((pos_pred - self._target_pos) ** 2), dim=-1) 
        met = self.metric_reward_fn(squared_distance, "pos")
        return - squared_distance, met # maximization of reward coincide with the minimization of the distance

    def expl_reward_fn(self, seq):
        # computation of intrinsic reward
        inp_feat = seq["stoch"].flatten(-2) if self.stoch_only else seq["feat"]
        
        obj_onehot = torch.eye(
            self.obj_instances + 1, device=inp_feat.device
        ).repeat(*inp_feat.shape[:2], 1, 1)

        x, _ = self.wm.heads["object_decoder"]._object_latent_extractor(
            inp_feat, obj_onehot
        )

        rw_intr = 0
        if self.reward_coeff["rw_intr"] > 0:
            for i in range(self.obj_instances):
                rw_intr += self.compute_intr_reward(x["mean"][:, :, i])
        # rw_intr = self.compute_intr_reward(obj_pos) intrinsic reward computation over object positon in space

        rw_task = torch.zeros_like(rw_intr)
        if self.reward_coeff["rw_task"] > 0:
            rw_task = self.wm.heads["reward"](seq["feat"]).mean

        # output final results
        self.rw_dict = {
            "rw_intr": rw_intr,
            "rw_task": rw_task,
        }
        rw_norm = {}
        mets = {}

        for key, val in self.rw_dict.items():
            rw_norm[key], met = self.rewnorm_dict[key](val)
            met = {f"{key}_{k}": v for k, v in met.items()}
            mets.update(met)

        rw = 0
        for key, val in rw_norm.items():
            rw = rw + self.reward_coeff[key] * val

        return rw, mets

    def report(self, data):
        report = {}
        text = {}
        
        data = self.wm.preprocess(data)
        with torch.no_grad():
            for key in self.wm.heads["decoder"].cnn_keys:
                name = key.replace("/", "_")
                report[f"{name}"] = self.wm.video_pred(data, key, "decoder")

            for key in self.wm.heads["object_decoder"].cnn_keys:
                name = key.replace("/", "_")
                report[f"{name}"] = self.wm.video_pred(
                    data, key, "object_decoder", nvid=2)

        return report, text

    def init_from(self, other):
        super(FocusAgent, self).init_from()
        
        # copy object_decoder parameters
        utils.hard_update_params(
            other.wm.heads["object_decoder"], self.wm.heads["object_decoder"]
        )
        
class OCWorldModel(WorldModel):
    # World Model between dreamer and focus needs to be aligned in processing of the input
    def __init__(self, config, obs_space, act_dim, tfstep):
        super().__init__(config, obs_space, act_dim, tfstep)
        self.heads["object_decoder"] = common.ObjDecoder(
            self.shapes, **self.cfg.object_decoder, embed_dim=self.inp_size, symlog_outputs=self.full_cfg.agent.symlog_inputs, obj_extractor_cfg=self.cfg.object_extractor)
        
        if self.cfg.get("object_encoder", False):
            self.object_encoder = common.ObjEncoder(self.shapes, symlog_inputs=self.full_cfg.agent.symlog_inputs, objlatent_ratio=self.cfg.objlatent_ratio, **self.cfg.object_encoder)
            self._shape_skill_latent = int(32 * self.cfg.object_decoder.cnn_depth * self.cfg.objlatent_ratio)
        
        self.model_init()
        self.stoch_only = not self.cfg.rssm.full_posterior

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
        assert len(kl_loss.shape) == 0 or (
            len(kl_loss.shape) == 1 and kl_loss.shape[0] == 1
        ), kl_loss.shape
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        obj_feat = post["stoch"].flatten(-2) if self.stoch_only else feat
        
        obj_states = {}
        for name, head in self.heads.items():
            grad_head = name in self.grad_heads
            inp = feat if name != "object_decoder" else obj_feat # add on to test if performance of object centric decoder improves only with positional information and not the dynamic one
            inp = inp if grad_head else stop_gradient(inp)

            out = (
                head(inp, masks=data["segmentation"])
                if name == "object_decoder"
                else head(inp)
            )

            if name == "object_decoder":
                obj_states["post"] = out.pop("post")
                
            dists = out if isinstance(out, dict) else {name: out}
            
            for key, dist in dists.items():
                like = 0
                
                # handled differently with respects to parent class, needs to be separated per instance
                if key == "segmentation":
                    seg = data[key].permute(0, 1, 3, 4, 2)
                    like = dist.log_prob(seg)
                elif key == "rgb" or key == "depth":
                    instances_dim = dist.mean.shape[2]
                    images = data[key].unsqueeze(2).repeat(1, 1, instances_dim, 1, 1, 1)
                    like = dist.log_prob(images)
                else:
                    like = dist.log_prob(data[key])

                likes[key] = like
                losses[key] = -like.mean()
            
        if self.cfg.get("object_encoder", False):
            obj_states["prior"] = self.object_encoder(data["objects_pos"])["prior"]
            del obj_states["post"]["dist"] # remove dist to pick only one object
            del obj_states["prior"]["dist"] # remove dist to pick only one object
            
            obj_states["post"] = {k: v[:,:,0,:self._shape_skill_latent].unsqueeze(2) for k, v in obj_states["post"].items()} # consider only first object in the scene
            obj_states["prior"] = {k: v[:,:,0,:self._shape_skill_latent].unsqueeze(2) for k, v in obj_states["prior"].items()} # consider only first object in the scene
            
            # reduce the dimension of the object states to consider for matching purposes, 
            # loss type determined by distance_mode
            loss_fn = getattr(common.DistLosses, f"{self.object_encoder.distance_mode}_loss")  
            losses["pose_prior"] = loss_fn(obj_states["post"], obj_states["prior"], self.cfg.objEnc_MSE_ratio)

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
        # decoding of the object_position for reporting
        decoder = self.heads[head]

        truth = data[key][:nvid][0].unsqueeze(1)

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

    def video_pred(self, data, key, head, nvid=4):
        if key == "rgb" or key == "depth":
            decoder = self.heads[head]  # B, T, C, H, W
            truth = data[key][:nvid] + 0.5
            embed = self.encoder(data)
            states, _ = self.rssm.observe(
                embed[:nvid, :5],
                data["action"][:nvid, :5],
                data["is_first"][:nvid, :5],
            )
            
            feat = states["stoch"].flatten(-2) if self.stoch_only else self.rssm.get_feat(states)

            recon = decoder(
                feat, data["segmentation"][:nvid, :5]
            )[key].mean[:nvid]

            recon_unmasked = decoder(feat)[key].mean[:nvid]

            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.rssm.imagine(data["action"][:nvid, 5:], init)
            prior_feat = prior["stoch"].flatten(-2) if self.stoch_only else self.rssm.get_feat(prior)
            prior_recon = decoder(
                prior_feat, data["segmentation"][:nvid, 5:]
            )[key].mean

            prior_recon_unmasked = decoder(prior_feat)[key].mean

            model = torch.clip(
                torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1
            )

            model_unmasked = torch.clip(
                torch.cat([recon_unmasked[:, :5] + 0.5, prior_recon_unmasked + 0.5], 1),
                0,
                1,
            )

            # create masks for truth
            masks = data["segmentation"]
            chs = truth.shape[2]
            for m in range(masks.shape[2]):
                if m == 0:
                    truth_out = (
                        masks[:nvid, :, m].unsqueeze(2).repeat(1, 1, chs, 1, 1) * truth
                    )
                else:
                    temp = (
                        masks[:nvid, :, m].unsqueeze(2).repeat(1, 1, chs, 1, 1) * truth
                    )
                    truth_out = torch.cat((truth_out, temp), 4)

            # divide model output
            model = model.permute(0, 1, 3, 4, 2, 5).reshape(
                *model.shape[:2],
                *model.shape[3:-1],
                model.shape[2] * model.shape[-1],
            )
            model_unmasked = model_unmasked.permute(0, 1, 3, 4, 2, 5).reshape(
                *model_unmasked.shape[:2],
                *model_unmasked.shape[3:-1],
                model_unmasked.shape[2] * model_unmasked.shape[-1],
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
            
            feat = states["stoch"].flatten(-2) if self.stoch_only else self.rssm.get_feat(states)

            recon = decoder(feat)[key].mean[:nvid]  # mode

            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.rssm.imagine(data["action"][:nvid, 5:], init)
            prior_feat = prior["stoch"].flatten(-2) if self.stoch_only else self.rssm.get_feat(prior)
            
            prior_recon = decoder(prior_feat)[key].mean  # mode

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
