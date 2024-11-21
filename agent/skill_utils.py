import numpy as np
import torch
import torch.nn.functional as F
import agent.dreamer_utils as common
import agent.utils as agent_utils

import utils
from agent.utils import stop_gradient

def get_feat_ac(seq):
  return torch.cat([seq['feat'], seq['skill']], dim=-1) 

class SkillActorCritic(common.Module):
  def __init__(self, config, act_spec, tfstep, skill_dim, solved_meta=None, sampling_strategy='object_context_position'):
    super().__init__()
    self.cfg = config.agent
    self.act_spec = act_spec
    self.tfstep = tfstep
    self._use_amp = (config.precision == 16)
    self.hor = config.imag_horizon
    self.device = config.device
    
    self.sampling_strategy = sampling_strategy
    self.solved_meta = solved_meta
    self.skill_dim = skill_dim
    inp_size = self.cfg.world_model.rssm.deter 
    if self.cfg.world_model.rssm.discrete: 
      inp_size += self.cfg.world_model.rssm.stoch * self.cfg.world_model.rssm.discrete
    else:
      inp_size += self.cfg.world_model.rssm.stoch
    
    inp_size += skill_dim if type(skill_dim) == int else sum(skill_dim)
    self.actor = agent_utils.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = agent_utils.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = agent_utils.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0 
    else:
      self._target_critic = self.critic
    self.actor_opt = agent_utils.Optimizer('skill_actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = agent_utils.Optimizer('skill_critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = agent_utils.StreamNorm(**self.cfg.skill_reward_norm, device=self.device)

  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    with agent_utils.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        B,T , _ = start['deter'].shape
        if self.solved_meta is not None:
          if self.solved_meta['skill'].shape[0] == 1: # in case a single goal is provided repeat for the amount of expected dimensions
            if type(self.solved_meta['skill']) == np.ndarray:
              img_skill = torch.from_numpy(self.solved_meta['skill'])[0,0].repeat(B*T, 1).to(self.device)
            else:
              img_skill = self.solved_meta['skill'][0,0].repeat(B*T, 1).to(self.device) # target for the agent
          else: 
            img_skill = self.solved_meta['skill'].reshape(B*T, -1) # different goal for batch size, but same for batch length
        
        else:
          if self.sampling_strategy == 'discrete':
            img_skill = F.one_hot(torch.randint(0, self.skill_dim, 
                                        size=(B*T,), device=self.device), num_classes=self.skill_dim).float()
          elif self.sampling_strategy == 'continuous':
            img_skill = torch.randn((B*T, self.skill_dim), device=self.device)
            img_skill = img_skill / torch.norm(img_skill, dim=-1, keepdim=True)
          elif self.sampling_strategy == 'object_context_position':
            # NOTE: this assumes background is always the last slot
            obj_dim, pos_dim = self.skill_dim[0], self.skill_dim[1]
            img_obj = F.one_hot(torch.randint(0, obj_dim - 1, size=(B*T,), device=self.device), num_classes=obj_dim).float()
            img_pos = torch.rand((B*T, pos_dim), device=self.device) * 2 - 1.
            img_skill = torch.cat([img_obj, img_pos], dim=-1)

        seq = world_model.imagine(self.actor, start, is_terminal, self.hor, task_cond=img_skill)
        
        seq['skill'] = seq.pop('task')
        reward, _ = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'skill_reward_{k}': v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with agent_utils.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):

        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)

        # start = {k: stop_gradient(v.transpose(0,1)) for k,v in start.items()}
        # start_target, _ = self.target(start)
        # critic_loss_start, _ = self.critic_loss(start, start_target)
        # critic_loss += critic_loss_start
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))

    if world_model.cfg.name == "lexa" and world_model.full_cfg.agent.distance_mode == "temporal":
      with agent_utils.RequiresGrad(self.dynamical_distance):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            inp_embed = world_model.heads["embed"](seq["feat"]).mode
            dd_loss = self.get_dynamical_distance_loss(self, world_model, inp_embed)
        metrics.update(self.dd_opt(dd_loss, self.dynamical_distance.parameters()))
            
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): #, step):
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(stop_gradient(get_feat_ac(seq)[:-2]))
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean # .mode()
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    elif self.cfg.actor_grad == 'both':
      baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean # .mode()
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['skill_actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:,:,None]
    ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean() 
    metrics['skill_actor_ent'] = ent.mean()
    metrics['skill_actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(get_feat_ac(seq)[:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'skill_critic': dist.mean.mean() } # .mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward'] 
    disc = seq['discount'] 
    value = self._target_critic(get_feat_ac(seq)).mean #.mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['skill_critic_slow'] = value.mean()
    metrics['skill_critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1 # .assign_add(1)
