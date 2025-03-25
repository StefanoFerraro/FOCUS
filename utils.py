import math
import random
import re
import time
from functools import wraps
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from typing import Union
import skimage

import hydra 

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            try:
                self.prev_states.append(model.training)
                model.train(False)
            except:
                pass

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type
    )
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]), norm_type
    )
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def retry(func):
    """
    A Decorator to retry a function for a certain amount of attempts
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except (OSError, PermissionError):
                attempts += 1
                time.sleep(0.1)
        raise OSError("Retry failed")

    return wrapper


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RMS(object):
    """running mean and std"""

    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + torch.square(delta) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PBE(object):
    """particle-based entropy based on knn normalized by running mean"""

    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep, cdist=False, apply_log=True):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        if cdist:
            sim_matrix = torch.cdist(source, target.detach(), p=2)
        else:
            sim_matrix = torch.norm(
                source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
                dim=-1,
                p=2,
            )
        reward, _ = sim_matrix.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = (
                torch.maximum(
                    reward - self.knn_clip, torch.zeros_like(reward, device=self.device)
                )
                if self.knn_clip >= 0.0
                else reward
            )  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = (
                torch.maximum(
                    reward - self.knn_clip, torch.zeros_like(reward, device=self.device)
                )
                if self.knn_clip >= 0.0
                else reward
            )
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        if apply_log:
            reward = torch.log(reward + 1.0)
        return reward

def make_dreamer_agent(obs_space, action_spec, cfg):
    from copy import deepcopy

    agent_cfg = cfg.agent
    cfg = deepcopy(cfg)
    
    del cfg.agent
    return hydra.utils.instantiate(
        agent_cfg,
        cfg=cfg,
        obs_space=obs_space,
        act_spec=action_spec,
    )

def TSNE_analysis(self):
    # plot tsne of latent space for a fix trajectory, to see how the latent space is evolving
    feat = []
    stoch_feat = []
    deter_feat = []
    obj_feat = []
    rgb_video = []
    video_length = 100
    video = np.empty([1, video_length + 1 , 3, *self.cfg.env.renderer.size])
    obs = self.eval_env.reset()
    rescale = lambda x: scipy.ndimage.zoom(x, (4, 4, 1), order=0) 
    rgb_video.append(rescale(obs["rgb"].transpose(1,2,0)))

    meta = self.agent.init_meta()
    step = 0
    agent_state = None
    eval_mode = True

    # start-up agent with one env episode
    with torch.no_grad():
        while step < video_length:
            # obs = torch.from_numpy(obs["rgb"].copy()).unsqueeze(0).float().to(agent.device)
            tensor_obs = {
                k: torch.as_tensor(np.copy(v), device=self.agent.device).unsqueeze(0)
                for k, v in obs.items()
            }

            f = self.agent.wm.encoder(self.agent.wm.preprocess(tensor_obs))
            feat.append(f.cpu().numpy()[0])
            
            action, agent_state = self.agent.act(
                                    obs,
                                    meta,
                                    step,
                                    eval_mode=eval_mode,
                                    state=agent_state,
                                )
            
            obs = self.eval_env.step(action)
            rgb_video.append(rescale(obs["rgb"].transpose(1,2,0)))
            
            # agent_state = (agent_state[0], torch.tensor(pos))
            # should update the agent state with the action that lead to the next state, otherwise, 
            # rssm expects to have a continuity betweeen consenquent state|action pairs
            
            s_f = agent_state[0]["stoch"].flatten(-2).unsqueeze(0)
            stoch_feat.append(s_f.cpu().numpy()[0])
            
            d_f = agent_state[0]["deter"].unsqueeze(0)
            deter_feat.append(d_f.cpu().numpy()[0])
            
            if self.agent.name == "skill_focus":
                f_i = s_f if not self.cfg.agent.world_model.rssm.full_posterior else self.agent.wm.rssm.get_feat(agent_state[0]).unsqueeze(0)
                o_f = self.agent.wm.heads["object_decoder"].object_latent_extractor(f_i)["post"]["mean"]
                obj_feat.append(o_f.cpu().numpy()[0])
            step += 1
    
    feat, stoch_feat, deter_feat = np.array(feat), np.array(stoch_feat)[:,0], np.array(deter_feat)[:,0]
    feat_dict = {"enc_feat": feat, "stoch_feat":stoch_feat, "deter_feat":deter_feat,}
    if self.agent.name == "skill_focus": 
        obj_feat = np.array(obj_feat)[:,0,0]
        feat_dict["obj_feat"] = obj_feat
    TSNE_analysis = {}
    
    rgb_video = np.array(rgb_video)
    
    for k, f in feat_dict.items():
        if len(f.shape) < 2:
            continue
        TSNE_analysis[k] = TSNE(n_components=2).fit_transform(f)

    fig, axs = plt.subplots(1, len(TSNE_analysis), figsize=(3 * len(TSNE_analysis), 3)) #, subplot_kw=dict(projection='3d'))
    for i, k in enumerate(TSNE_analysis.keys()):
        axs[i].set_title(k)

    color = np.linspace(0, 1, video_length)
    for step, rgb in enumerate(rgb_video):
        # concatenate rgb image with tsne plot of eqivalent step, color evolve over steps
        for i, v in enumerate(TSNE_analysis.values()):
            axs[i].scatter(v[:step,0], v[:step,1], c=color[:step])

        fig.canvas.draw()
        data = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)[:,:,:3] # keep only rgb channels
        data = cv2.resize(data, dsize=(256 * len(TSNE_analysis), 256), interpolation=cv2.INTER_CUBIC) # resize to fit rgb image

        if step==0:
            video = np.expand_dims(np.hstack([rgb, data]), axis=0)    
        else:
            curr_frame = np.expand_dims(np.hstack([rgb, data]), axis=0)
            video = np.concatenate([video, curr_frame], axis=0)    
    
    video = np.uint8(np.expand_dims(video.transpose(0,3,1,2), axis=0) * 255)
    self.logger.log_video({'TSNE_video' : video }, self.global_frame) 

class DummyProfiler(object):
    def __init__(self, **kwargs):
        pass
    
    def step(self, *args):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass
    
class Profiler(DummyProfiler):
    def __init__(self, warmup, active_for):
        self.warmup = warmup
        self.active_for = active_for
        self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=self.warmup, active=self.active_for, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/mnt/home/focus/log/skill_focus'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
    
    def step(self, global_step: int) -> Any:
        self.profiler.step()
        if global_step == self.warmup + self.active_for:
            self.profiler.stop()
            
def object_metrics(in_areas, cumm_pos_disp, cumm_ang_disp, vert_pos_disp, episode_frame):
    placement = {"left_placement": 0, "right_placement": 1, "close_placement": 2, "far_placement": 3, "up_placement": 4}
    placement = {k:float(in_areas[v] / episode_frame) for k,v in placement.items()}
    displacement = {"pos_displacement":cumm_pos_disp, "ang_displacement":cumm_ang_disp, "vertical_displacement": vert_pos_disp}
    metrics = {}
    metrics.update(placement)
    metrics.update(displacement)
    
    return metrics
        
def move_to_target_metrics(obj_pos, target_pos):
    metrics = {}
    # exponential distance from the target at the end of episode
    metrics["move_to_target_final"] = np.exp(- np.linalg.norm( obj_pos[-1] - target_pos) / np.linalg.norm(target_pos))
    # exponential min distance to target during the entire episode
    metrics["move_to_target_min"] = np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).max()
    # exponential max distance to target during the entire episode
    metrics["move_to_target_max"] = np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).min()
    # exponential max distance to target during the entire episode
    metrics["move_to_target_mean"] = np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).mean()
    # episode average number of pixels for main object segmentation mask
    metrics["final_distance_to_target"] = - np.linalg.norm(obj_pos[-1] - target_pos, axis=-1) 
    metrics["min_distance_to_target"] = - np.linalg.norm(obj_pos[-1] - target_pos, axis=-1).min() 
    
    return metrics

def log_metrics_dict(metrics, log_fn):
    for k,v in metrics.items():
        log_fn(k, v)
        
def init_metrics_counters(self):
    self.contact_count = 0
    self.metrics = None
    self.in_areas = np.array([0, 0, 0, 0, 0])
    self.obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
    self.cumm_pos_displacement = 0
    self.cumm_ang_displacement = 0
    self.cumm_vertical_displacement = 0
    self.segmentation_obj_pixels = 0
    
    self.episode_step = 0
    self.step_to_success = self._horizon
    self.episode_reward = 0
    
def update_metrics_counters(self, obs):
    self.episode_reward += obs["reward"]
    self.episode_step += 1
    # Hacky way to say that step_to_success was set
    if self.step_to_success == self._horizon and obs["success"]:
        self.step_to_success = self.episode_step
    self._global_step += 1
    self.contact_count += obs["contact"]
    self.in_areas += np.array(obs["in_areas"])
    self.obj_pos = np.concatenate((self.obj_pos, [obs["objects_pos"][0]])) 
    self.cumm_pos_displacement += obs["pos_displacement"]
    self.cumm_ang_displacement += obs["ang_displacement"]
    self.cumm_vertical_displacement += obs["vertical_displacement"]
    self.segmentation_obj_pixels += np.sum(obs["segmentation"][0])

def custom_target_2d(obs, target_pos, diam, env):
    if diam == 1:
        target_pixels = [np.array(x * 100 + 32) for x in target_pos[::-1]] # single pixel target
    else:
        target_pixels = skimage.draw.disk((target_pos[::-1] * [-100, 100] + [32, 32]), radius=int(diam/2))
    
    torch_to_numpy = lambda x: x.detach().cpu().numpy() if type(x) == torch.Tensor else x

    obs["rgb"][... ,0,slice(*target_pixels)] = 255
    obs["rgb"][... ,1,slice(*target_pixels)] = 0
    obs["rgb"][... ,2,slice(*target_pixels)] = 0
    # adapt proprio and reward to the new target
    coord_dist_to_target = torch_to_numpy(obs["objects_pos"][...,0,:]) - target_pos
    dist_to_target = np.sqrt(np.sum((coord_dist_to_target)**2, axis=-1))
    rewards = env.get_reward(dist_to_target)
    
    # provide directly the target position in proprio and not the distance between target and eef
    obs["proprio"][...,2:4] = torch.tensor(target_pos, device="cuda") if type(obs["proprio"]) == torch.Tensor else target_pos
    obs["reward"] = torch.tensor(rewards, device="cuda").unsqueeze(-1) if type(obs["reward"]) == torch.Tensor else rewards
                    
    return obs

#colorblind color palette okabeito
def colors():
    lightblue = "#56B4E9"
    yellow = "#F0E442"
    orange = "#E69F00"
    green = "#009E73"
    purple = "#CC79A7"
    red = "#D55E00"
    blue = "#0072B2"
    black = "#000000"
    
    return [orange, lightblue, green, purple, red, yellow, blue, black]

def dist_as_reward(obs, task):
    if task in ["reacher_easy", "reacher_hard"]:
        obs["reward"] = - torch.sqrt(torch.sum(obs["proprio"][:,:,2:4]**2, dim=-1)).unsqueeze(-1)
        obs["target"] = obs["proprio"][:,:,2:4] # isolate target value
        obs["proprio"] = torch.cat((obs["proprio"][:,:,:2], obs["proprio"][:,:,4:]), dim=-1)
    elif task in ["Lift", "shelf-place", "bin-picking"]:
        obs["reward"] = - torch.linalg.norm(obs["proprio"][:,:,-3:], dim=-1).unsqueeze(-1)
        obs["target"] = obs["proprio"][:,:,-3:] # isolate target value
        obs["proprio"] = obs["proprio"][:,:,:-3]
    else:
        raise ValueError("Task not implemented")
    return obs

def exp_func(agent, x, modulation_factor):
    init_low_bea = agent.init_lower_bound_expl_area 
    init_up_bea = agent.init_upper_bound_expl_area
    min_ea = agent.min_exploration_area
    max_ea = agent.max_exploration_area
    # clipping of lower and maximum values
    lower_expl_area = np.clip((np.exp(x / modulation_factor) * init_low_bea), min_ea, max_ea)
    upper_expl_area = np.clip((np.exp(x / modulation_factor) * init_up_bea), min_ea, max_ea)
    return [lower_expl_area, upper_expl_area]

def expl_area_update(agent, global_step, modulation_factor=10e6, curriculum_learning=True):        
    if curriculum_learning:
        new_exploration_area = exp_func(global_step, modulation_factor)
    else:
        # sample from full exploration area
        new_exploration_area = [agent.min_exploration_area, agent.max_exploration_area]
    
    agent.set_exploration_area(new_exploration_area)
    return new_exploration_area

def generate_target(exploration_limits : list,  curriculum_learning: bool, global_step: int, modulation_factor: int, sampling_stategy: str = "uniform", shape: Union[list, np.ndarray] = [1]):
    
    if curriculum_learning:
        exploration_area = exp_func(global_step, modulation_factor)
    else:
        # sample from full exploration area
        exploration_area = [exploration_limits[0], exploration_limits[1]]
    
    min = np.tile(exploration_area[0], shape)
    max = np.tile(exploration_area[1], shape)
    mean = np.tile(np.mean(exploration_area, axis=0), shape)
    sigma = np.tile((np.array(exploration_area[1]) - np.array(exploration_area[0])) / 6, shape) 
    if sampling_stategy == "uniform":
        target = np.random.uniform(min, max)    
        
    elif sampling_stategy == "normal":
        target = np.random.normal(mean, sigma)
        target = np.clip(target, min, max)
    
    return target

def get_target(cfg, env, replay_storage, replay_iter, global_step):
    if cfg.env_target: # target directlly from environment
        target = target_pos = env.get_target()
    else: # target set from: 
        target_obs_mode = cfg.only_obs_target or (cfg.mixed_target and random.choice([True, False]))                
        
        if cfg.target_from_replay_bf:  # Sampling goals from replay buffer 
            replay_storage._length = 1
            target_obs = next(replay_iter)
            replay_storage._length = cfg.batch_length # Restoring replay buffer settings
            if not cfg.batch_sampling:
                target_obs = {k: v[0].unsqueeze(0) for k,v in target_obs.items()} # batch size cannot be modified after initialization, so take only the first element
            target_pos = target_obs["objects_pos"][:,:,0].cpu().numpy()
        else: # Sampling positions uniformelly from the workspace
            shape = [cfg.batch_size, 1, 1] if cfg.batch_sampling else [1,1,1]
            target_pos = generate_target(env.limits_exploration_area, cfg.curriculum_learning, global_step, cfg.target_modulator, cfg.target_sampling_generation_strategy, shape=shape)
            
            if target_obs_mode: # obtain observation (skip computation if not required) 
                target_obs = env.set_goal_state(target_pos[0,0])    
                if target_obs["is_last"] == True: env.reset()
                target_obs = { k: torch.as_tensor(np.copy(v), device=cfg.device).unsqueeze(0).unsqueeze(0) for k, v in target_obs.items()}
            
    target = target_obs if target_obs_mode else target_pos
    
    return target, target_pos