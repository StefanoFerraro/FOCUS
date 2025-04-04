import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from agent.mf_utils import *
from agent.dreamer_utils import *
import utils

def iql_loss(diff, expectile=0.9):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLAgent:
    def __init__(self,
                 name,
                 cfg, 
                 obs_space, 
                 act_spec, 
                 device,
                 lr,
                 hidden_dim,
                 feature_dim,
                 target_tau,
                 distributional,
                 temperature,
                 expectile,
                 obs_keys,
                 drq_encoder,
                 drq_aug,):
        self.cfg = cfg
        self.name = name
        frame_stack = cfg.frame_stack
        self.act_spec = act_spec
        self.obs_space = obs_space
        self.action_dim = np.prod(act_spec.shape)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = target_tau
        self.temperature = temperature
        self.expectile = expectile
        self.obs_keys = obs_keys.split('|')
        shapes = {}
        for k,v in obs_space.items():
            shapes[k] = list(v.shape)
            if len(v.shape) == 3:
                shapes[k][0] = shapes[k][0] * frame_stack
        
        self.frame_stack = frame_stack
        self.obs_buffer = defaultdict(list) 
        self._batch_reward = 0

        # models
        self.encoders = {}
        self.augs = {}
        for k in self.obs_keys:
            if len(shapes[k]) == 3:
                img_size = shapes[k][-1]
                pad = img_size // 21 # pad=4 for 84
                self.augs[k] = utils.RandomShiftsAug(pad=pad) if drq_aug else nn.Identity()
                if drq_encoder:
                    self.encoders[k] = DrQEncoder(shapes[k], key=k, is_rgb=obs_space[k].shape[0] == 3).to(self.device)
                else:
                    self.encoders[k] = DreamerEncoder(shapes[k], key=k, is_rgb=obs_space[k].shape[0] == 3).to(self.device)
            else:
                self.augs[k] = nn.Identity()
                self.encoders[k] = IdentityEncoder(k)
                self.encoders[k].repr_dim = shapes[k][0]
        self.encoders = nn.ModuleDict(self.encoders)
        self.enc_repr_dim = sum(e.repr_dim for e in self.encoders.values())

        self.actor = ActorFixedStd(self.enc_repr_dim, self.action_dim, hidden_dim, feature_dim).to(device)

        if distributional:
            self.critic = DistributionalCritic(self.enc_repr_dim, self.action_dim,
                                hidden_dim, feature_dim).to(device)
            self.critic_target = DistributionalCritic(self.enc_repr_dim, self.action_dim,
                                        hidden_dim, feature_dim).to(device)
        else:
            self.critic = Critic(self.enc_repr_dim, self.action_dim,
                                hidden_dim, feature_dim).to(device)
            self.critic_target = Critic(self.enc_repr_dim, self.action_dim,
                                        hidden_dim, feature_dim).to(device)

        self.value = ValueCritic(self.enc_repr_dim, hidden_dim, feature_dim).to(device)

        # Target nets    
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.encoder_opt = torch.optim.Adam(self.encoders.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)

        # schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=int(cfg.num_train_frames))

        self.train()
        self.critic_target.train()

    def init_meta(self):
        return {}

    def get_meta_specs(self):
        return {}

    def update_meta(self, meta, global_step, time_step):
        return self.init_meta()

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def act(self, obs, meta, step, eval_mode, state,):
        is_first = np.all(obs['is_first']) or len(self.obs_buffer[self.obs_keys[0]]) == 0
        
        # TODO: the frame stack buffer should not be in the agent!
        for k in self.obs_keys:
            obs[k] = torch.as_tensor(np.copy(obs[k]), device=self.device).unsqueeze(0)
            # States do not require frame stacking
            if len(obs[k].shape) == 2:
                continue

            if is_first:
                self.obs_buffer[k] = [obs[k]] * self.frame_stack
            else:
                self.obs_buffer[k].pop(0)
                self.obs_buffer[k].append(obs[k])
            obs_ch = obs[k].shape[1]
            obs_size = obs[k].shape[2:]
            obs[k] = torch.stack(self.obs_buffer[k], dim=1).reshape(-1, obs_ch * self.frame_stack, *obs_size)
        
        obs = torch.cat([ e(e.preprocess(obs)) for e in self.encoders.values()], dim=-1)
        
        policy = self.actor(obs, 0)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
            if step < (self.cfg.num_seed_frames // self.cfg.action_repeat):
                action.uniform_(-1.0, 1.0)
        # @returns: action, state
        return action.squeeze(0).cpu().numpy(), None

    def update_value(self, obs, action, step):
        metrics = dict()

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(obs, action)
            if getattr(self.critic, 'distributional', False):
                target_Q1, target_Q2 = from_categorical(target_Q1), from_categorical(target_Q2)
            target_V = torch.min(target_Q1, target_Q2)

        V = self.value(obs)
        value_loss = iql_loss(target_V - V, self.expectile).mean()

        metrics['target_V'] = target_V.mean().item()
        metrics['value_V'] = V.mean().item()
        metrics['value_loss'] = value_loss.item()

        # optimize encoder and value
        self.value_opt.zero_grad(set_to_none=True)
        # self.encoder_opt.zero_grad(set_to_none=True)
        
        value_loss.backward()

        value_grad_norm = nn.utils.clip_grad_norm(self.value.parameters(), np.inf)
        # enc_grad_norm = nn.utils.clip_grad_norm(self.encoders.parameters(), np.inf)
        
        self.value_opt.step()
        # self.encoder_opt.step()

        metrics['critic_grad_norm'] = value_grad_norm.item()
        # metrics['encoder_grad_norm'] = enc_grad_norm.item()
        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            target_V = self.value(next_obs)
            target_Q = reward + (discount * target_V)
            if getattr(self.critic, 'distributional', False):
                target_Q_dist = to_categorical(target_Q,)

        Q1, Q2 = self.critic(obs, action)
        if getattr(self.critic, 'distributional', False):
            critic_loss = - torch.mean(torch.sum(torch.log_softmax(Q1, -1) * target_Q_dist.squeeze(-1).detach(), -1)) - torch.mean(torch.sum(torch.log_softmax(Q2, -1) * target_Q_dist.squeeze(-1).detach(), -1))
            Q1, Q2 = from_categorical(Q1), from_categorical(Q2)
        else:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        critic_grad_norm = nn.utils.clip_grad_norm(self.critic.parameters(), np.inf)
        enc_grad_norm = nn.utils.clip_grad_norm(self.encoders.parameters(), np.inf)
        
        self.critic_opt.step()
        self.encoder_opt.step()

        metrics['critic_grad_norm'] = critic_grad_norm.item()
        metrics['encoder_grad_norm'] = enc_grad_norm.item()
        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        with torch.no_grad():
            v = self.value(obs)
            q1, q2 = self.critic_target(obs, action)
            q = torch.minimum(q1, q2)
            exp_a = torch.exp((q - v) * self.temperature)
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        mu = self.actor(obs, 0).mean
        actor_loss = (exp_a.unsqueeze(-1) * ((mu - action)**2)).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm(self.actor.parameters(), np.inf)
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_mean_stddev'] = self.actor._mu_std
        metrics['actor_adv'] = (q-v).mean()
        metrics['actor_grad_norm'] = actor_grad_norm.item()
        return metrics

    def update(self, batch, step, only_actor=False, only_critic=False, *args, **kwargs):
        assert sum([only_actor, only_critic]) <= 1
        metrics = dict()

        obs, next_obs = {}, {}
        for k in self.obs_keys:
            b, t = batch[k].shape[:2]
            obs_ch = batch[k].shape[2]
            obs_size = batch[k].shape[3:]
            if len(obs_size) == 2:
                obs[k] = batch[k][:, 0:self.frame_stack].reshape(b, obs_ch * self.frame_stack, *obs_size)
                next_obs[k] = batch[k][:, 1:self.frame_stack+1].reshape(b, obs_ch * self.frame_stack, *obs_size)
            else:
                obs[k] = batch[k][:, self.frame_stack-1].reshape(b, obs_ch, *obs_size)
                next_obs[k] = batch[k][:, self.frame_stack].reshape(b, obs_ch, *obs_size)
            obs[k] = self.augs[k](obs[k].float()).to(self.device)
            next_obs[k] = self.augs[k](next_obs[k].float()).to(self.device)

        action = batch['action'][:, self.frame_stack].to(self.device)
        reward = globals()[self.cfg.acting_reward_fn](batch, self.cfg.task)[:, self.frame_stack].to(self.device)
        discount = (batch['discount'][:, self.frame_stack] * self.cfg.discount).to(self.device)

        obs = torch.cat([e(e.preprocess(obs)) for e in self.encoders.values()], dim=-1)
        with torch.no_grad():
            next_obs = torch.cat([e(e.preprocess(next_obs)) for e in self.encoders.values()], dim=-1)

        metrics['batch_reward'] = reward.mean().item() 

        if not only_actor:
            # update critic
            metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
            # update value
            metrics.update(self.update_value(obs.detach(), action, step))
            # update target critic
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        if not only_critic:
            # update actor
            metrics.update(self.update_actor(obs.detach(), action, step))
        # @returns: state, metrics
        return None, metrics 