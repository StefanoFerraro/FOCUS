import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
import wandb
from dm_env import specs
import gym

from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ, PRIMAL_TASKS 
from env.utils import obs_specs

from env.make import make
import utils
from logger import Logger
from dreamer_replay import ReplayBuffer, make_replay_loader

# torch.backends.cudnn.benchmark = True

from torch.profiler import profile, record_function, ProfilerActivity

import warnings

warnings.filterwarnings("ignore")

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_dreamer_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy

    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(
        cfg,
        cfg=cur_config,
        obs_space=obs_space,
        act_spec=action_spec,
        is_finetune=cur_config.is_finetune,
    )


def segmentation_visualization(
    self,
    seg,
    palette=None,
):
    if palette is None:
        # Get random state before set seed, and restore random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(0, 255, size=(max(seg), 3))
        np.random.set_state(state)

    palette = np.array(palette)
    assert palette.shape[0] == max(seg)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    color_seg = np.zeros((seg.shape[1], seg.shape[2], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    return seg


class Workspace:
    def __init__(self, cfg, maindir=None, workdir=None):
        self.maindir = Path.cwd() if maindir is None else maindir
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f"workspace: {self.workdir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        cfg.agent.world_model.device = cfg.device

        # create logger
        self.logger = Logger(self.workdir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs
        domain = cfg.domain
        if cfg.agent.train_target_reach:
            cfg.env.target_modulator = cfg.target_modulator
        task = (
            cfg.task if cfg.task != "none" else PRIMAL_TASKS[self.cfg.domain]
        )  # -> which is the URLB default
        frame_stack = 1

        os.chdir(
            self.maindir
        )  # change to original working directory for loading URDF models

        # adapt object target pose center for specific task 
        # TODO create a function that takes care of it for all different cases
        # if self.cfg.task == "manipulator_bring_ball" or self.cfg.task == "manipulator_bring_peg":
        #     self.cfg.env.object_start_pos = [0.5, 0.32]
                
        self.train_env = make(
            domain,
            task,
            cfg.obs_type,
            frame_stack,
            cfg.action_repeat,
            cfg.seed,
            cfg.env,
        )

        self.eval_env = make(
            domain,
            task,
            cfg.obs_type,
            frame_stack,
            cfg.action_repeat,
            cfg.seed,
            cfg.env,
        )

        train_obs_spec = self.train_env.obs_space
        
        # if self.cfg.agent.name == "dreamer":
        #             # adapt object pos shape to single object
        #     train_obs_spec["objects_pos"] = gym.spaces.Box(
        #             -2, 2, (train_obs_spec["objects_pos"].shape[-1],), dtype=np.float32
        #         )

        os.chdir(self.maindir)

        objets_list = globals()[domain.upper() + "_TASKS_OBJ"][task]

        if cfg.agent.world_model.name == "focus":
            cfg.agent.world_model.objects = objets_list

        self.train_env.reset()
        # create agent
        self.agent = make_dreamer_agent(
            train_obs_spec,
            self.train_env.action_spec(),
            cfg,
            cfg.agent,
        )

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        data_specs = (
            *obs_specs(train_obs_spec),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBuffer(
            data_specs,
            meta_specs,
            self.workdir / "buffer",
            length=cfg.batch_length,
            **cfg.replay,
            device=cfg.device,
        )

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
        )

        self._replay_iter = None

        # Globals
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._horizon = cfg.env.horizon

    def reset(self, func):
        os.chdir(
          (self.maindir)
        )  # change to original working directory for loading URDF models
        obs = func.reset()
        os.chdir(self.workdir)

        return obs

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        if self.global_step == 0:
            return
        # To save time, we don't eval during training by default. Feel free to uncomment.
        # return
        step, episode, total_reward, total_success = 0, 0, 0, 0
        step_to_success = self._horizon
        step_to_success_list = []        

        # set to True to use task_behaviour, zero_shot performance
        self.agent.is_finetune = True

        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
        video = np.empty([1, int(self._horizon/2) + 1 , 3, *self.cfg.env.renderer.size])
        move_to_target_final, move_to_target_min, move_to_target_max, move_to_target_mean = [], [], [], []
        
        while eval_until_episode(episode):
            episode_data = []
            
            # set target before reset of env
            if self.cfg.agent.train_target_reach:
                # target = np.zeros_like(self.cfg.env.object_start_pos)
                # target[1] = self.target_update(self.cfg.env.target_modulator)[1]
                # pick random goal for evaluation
                target = self.eval_env.get_random_goal()[1]
                self.agent.set_target(target) # update pos target only along x axis 
                if self.cfg.env.visualize_target:   
                    self.eval_env.set_target(target)                    
    
            dreamer_obs = self.eval_env.reset()
            # dmc envs adapt objects position after the res
            if self.cfg.agent.train_target_reach and self.cfg.env.visualize_target:   
                self.eval_env.set_target(target)
                   
            dreamer_obs["eval_rgb"] = dreamer_obs["rgb"]
            
            episode_data.append(dreamer_obs)
            agent_state = None
            while not bool(dreamer_obs["is_last"]):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(
                        dreamer_obs,
                        meta,
                        self.global_step,
                        eval_mode=True,
                        state=agent_state,
                    )
                dreamer_obs = self.eval_env.step(action)
                
                # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
                
                if self.cfg.agent.train_target_reach and self.cfg.env.visualize_target:
                    dreamer_obs["eval_rgb"] = self.eval_env.get_rgb_with_target(target)

                episode_data.append(dreamer_obs)
                total_reward += dreamer_obs["reward"]
                step += 1
                # Hacky way to say that step_to_success was set
                if step_to_success == self._horizon and dreamer_obs["success"]:
                    step_to_success = (step * self.cfg.action_repeat) - (
                        episode * self._horizon
                    )
                # if self.agent.name == "dreamer":
                #     obj_pos = np.concatenate((obj_pos, [dreamer_obs["objects_pos"]]))
                # else:
                obj_pos = np.concatenate((obj_pos, [dreamer_obs["objects_pos"][0]]))

            # log specs
            if self.cfg.agent.train_target_reach:
                target_pos = self.agent._target_pos.cpu().numpy()
                move_to_target_final.append(np.exp(- np.linalg.norm(
                        obj_pos[-1] - target_pos)
                    / np.linalg.norm(target_pos))) # exponential distance from the target at the end of episode
                move_to_target_min.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1)
                    / np.linalg.norm(target_pos)).max()) # exponential min distance to target during the entire episode
                move_to_target_max.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1)
                    / np.linalg.norm(target_pos)).min()) # exponential max distance to target during the entire episode
                move_to_target_mean.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1)
                    / np.linalg.norm(
                        target_pos)).mean()) # exponential max distance to target during the entire episode
                
            if episode==0:
                video = np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)    
            else:
                video = np.concatenate([video, np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)], axis=-1)    
            
            episode += 1
            step_to_success_list += [step_to_success]
            step_to_success = self._horizon
            total_success += dreamer_obs["success"]
            obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 

        
        self.agent.is_finetune = False if not self.cfg.is_finetune else True
          
        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("avg_success", total_success / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("avg_step_to_success", sum(step_to_success_list) / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if self.cfg.agent.train_target_reach:
                log(
                    "move_to_target_final", np.mean(move_to_target_final))
                log(
                    "move_to_target_min", np.mean(move_to_target_min))
                log(
                    "move_to_target_max", np.mean(move_to_target_max))
                log(
                    "move_to_target_mean", np.mean(move_to_target_mean))

        # B, T, C, H, W = video.shape
        # last_video = np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data ], axis=0), axis=0)
        video = np.uint8(video * 255)
        self.logger.log_video({'eval_video' : video }, self.global_frame)

        # Eval episodes for testing the prior model (get fixed poses, decode the observation of the prior)
         
          
    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat
        should_train_step = utils.Every(
            train_every_n_steps * self.cfg.action_repeat,
            self.cfg.action_repeat,
        )
        should_log_scalars = utils.Every(
            self.cfg.log_every_frames, self.cfg.action_repeat
        )
        should_log_recon = utils.Every(
            self.cfg.recon_every_frames, self.cfg.action_repeat
        )

        episode_step, step_to_success, episode_reward = 0, self._horizon, 0
        try:
            _, dreamer_obs = self.reset(self.train_env)
        except:
            dreamer_obs = self.reset(self.train_env)
        
        # adapt dreamer use case for single object
        # if self.agent.name == "dreamer":
        #     dreamer_obs["objects_pos"] = dreamer_obs["objects_pos"][0]
            
        agent_state = None
        meta = self.agent.init_meta()
        data = dreamer_obs
        
          

        self.replay_storage.add(data, meta)
        metrics = None
        contact_count = 0
        in_areas = np.array([0, 0, 0, 0, 0])
        obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
   
        cumm_pos_displacement = 0
        cumm_ang_displacement = 0
        cumm_vertical_displacement = 0
        segmentation_obj_pixels = 0
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=490, active=10, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/mnt/home/focus/log/skill_focus'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
            # import time
        while train_until_step(self.global_step):
            # prof.step()
            # if self._global_step == 490:
            #     pass
            # if self._global_step == 500:
            #     prof.stop()
                
            if bool(dreamer_obs["is_last"]):
                self._global_episode += 1
                # if should_log_recon(self._global_step):
                #     if self.cfg.obs_type == 'pixels':
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                        log("success", dreamer_obs["success"])
                        log("step_to_success", step_to_success)
                        log("contact", float(contact_count / episode_frame))
                        # TODO: for the future, put everything in a dict and cycle through the keys
                        log(
                            "left_placement",
                            float(in_areas[0] / episode_frame),
                        )
                        log(
                            "right_placement",
                            float(in_areas[1] / episode_frame),
                        )
                        log(
                            "close_placement",
                            float(in_areas[2] / episode_frame),
                        )
                        log("far_placement", float(in_areas[3] / episode_frame))
                        log("up_placement", float(in_areas[4] / episode_frame))
                        log("pos_displacement", cumm_pos_displacement)
                        log("ang_displacement", cumm_ang_displacement)
                        log("vertical_displacement", cumm_vertical_displacement)
                        if self.cfg.agent.train_target_reach:
                            target_pos = self.agent._target_pos.cpu().numpy()
                            log(
                                "move_to_target_final",
                                np.exp(- np.linalg.norm(
                                    obj_pos[-1] - target_pos)
                                / np.linalg.norm(target_pos)) # exponential distance from the target at the end of episode
                            )
                            log(
                                "move_to_target_min",
                                np.exp(- np.linalg.norm(
                                    obj_pos - target_pos, axis=-1)
                                / np.linalg.norm(target_pos)).max() # exponential min distance to target during the entire episode
                            )
                            log(
                                "move_to_target_max",
                                np.exp(- np.linalg.norm(
                                    obj_pos - target_pos, axis=-1)
                                / np.linalg.norm(target_pos)).min() # exponential max distance to target during the entire episode
                            )
                            log(
                                "move_to_target_mean",
                                np.exp(- np.linalg.norm(
                                    obj_pos - target_pos, axis=-1)
                                / np.linalg.norm(
                                    target_pos)).mean() # exponential max distance to target during the entire episode
                            )
                            log(
                                "segmentation_obj_pixels",
                                float(segmentation_obj_pixels / episode_step)  # episode average number of pixels for main object segmentation mask
                            )
                        
                if self.cfg.agent.train_target_reach:
                    self.expl_area_update(self.cfg.env.target_modulator, self.cfg.curriculum_learning) # update pos target according to scheduler 
                    self.agent.update_target() # update target in the agent based on the new exploration area 
                    self.train_env.set_target(self.agent.get_target()[0,0,0].detach().cpu().numpy().copy())  # visually set the target                  
                
                contact_count = 0
                in_areas = np.array([0, 0, 0, 0, 0])
                obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 

                cumm_pos_displacement = 0
                cumm_ang_displacement = 0
                cumm_vertical_displacement = 0
                segmentation_obj_pixels = 0
                
                # save last model
                if self.global_step % 50000 == 0:
                    self.save_last_model()

                # reset env
                dreamer_obs = self.reset(self.train_env)
                # adapt dreamer use case for single object
                # if self.agent.name == "dreamer":
                #     dreamer_obs["objects_pos"] = dreamer_obs["objects_pos"][0]

                agent_state = None  # Resetting agent's latent state
                meta = self.agent.init_meta()
                data = dreamer_obs
                
                    
                self.replay_storage.add(data, meta)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                step_to_success = self._horizon

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time",
                    self.timer.total_time(),
                    self.global_frame,
                )
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, dreamer_obs)

            with torch.no_grad(), utils.eval_mode(self.agent):
                if seed_until_step(self.global_step):
                    action = self.train_env.act_space["action"].sample()
                else:
                    if self.cfg.agent.name=="skill_focus":
                        # In this case we want to have the skill agent acting 50% of the time, and the rest made by an exploratory agent
                        meta = {"use_skill_behaviour": self.global_episode % 2}
                    action, agent_state = self.agent.act(
                        dreamer_obs,
                        meta,
                        self.global_step,
                        eval_mode=False,
                        state=agent_state,
                    )
            
            # try to update the agent
            if not seed_until_step(self.global_step): # fill the replay buffer before training WM and agent
                if self.replay_storage._total_steps > 0:
                    if should_train_step(self.global_step):
                        metrics = self.agent.update(
                            next(self.replay_iter), self.global_step
                        )[1]
                    if should_log_scalars(self.global_step):
                        self.logger.log_metrics(metrics, self.global_frame, ty="train")
                    if self.global_step > 0 and should_log_recon(self.global_step):
                        videos, text = self.agent.report(next(self.replay_iter))

                        self.logger.log_video(videos, self.global_frame)
                        self.logger.log_text(text, self.global_frame)

            # take env step
            dreamer_obs = self.train_env.step(action)
            # adapt dreamer use case for single object
            # if self.agent.name == "dreamer":
            #     dreamer_obs["objects_pos"] = dreamer_obs["objects_pos"][0]


            episode_reward += dreamer_obs["reward"]
            data = dreamer_obs
            self.replay_storage.add(data, meta)

            # if self._global_step >= 4000:
            #     print(time.time() - t1)
            
            episode_step += 1
            # Hacky way to say that step_to_success was set
            if step_to_success == self._horizon and dreamer_obs["success"]:
                step_to_success = episode_step
            self._global_step += 1
            contact_count += dreamer_obs["contact"]
            in_areas += np.array(dreamer_obs["in_areas"])
            # if self.agent.name == "dreamer":
            #     obj_pos = np.concatenate((obj_pos, [dreamer_obs["objects_pos"]]))
            # else:
            obj_pos = np.concatenate((obj_pos, [dreamer_obs["objects_pos"][0]]))
                
            cumm_pos_displacement += dreamer_obs["pos_displacement"]
            cumm_ang_displacement += dreamer_obs["ang_displacement"]
            cumm_vertical_displacement += dreamer_obs["vertical_displacement"]
            segmentation_obj_pixels += np.sum(dreamer_obs["segmentation"][0])
        

    @utils.retry
    def save_snapshot(self):
        snapshot = self.get_snapshot_dir() / f"snapshot_{self.global_frame}.pt"
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = "_".join(
            [   "Pretrain",
                cfg.agent.name,
                cfg.env.name,
                cfg.task,
            ]
        )
        wandb.init(
            project=cfg.project_name,
            group=cfg.agent.name,
            name=exp_name,
        )
        wandb.config.update(dict(cfg))
        self.wandb_run_id = wandb.run.id

    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / "last_snapshot.pt"
        if snapshot.is_file():
            temp = Path(
                str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt")
            )
            os.replace(snapshot, temp)
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        if self.cfg.use_wandb:
            keys_to_save.append("wandb_run_id")
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def exp_func(self, x, modulation_factor):
        init_low_bea, init_up_bea, min_ea, max_ea = self.agent.init_lower_bound_expl_area, self.agent.init_upper_bound_expl_area, self.agent.min_exploration_area, self.agent.max_exploration_area
        # clipping of lower and maximum values
        lower_expl_area = np.clip((np.exp(x / modulation_factor) * init_low_bea), min_ea, max_ea)
        upper_expl_area = np.clip((np.exp(x / modulation_factor) * init_up_bea), min_ea, max_ea)
        return [lower_expl_area, upper_expl_area]

    def expl_area_update(self, modulation_factor=10e6, curriculum_learning=True):        
        if curriculum_learning:
            new_exploration_area = self.exp_func(self.global_step, modulation_factor)
        else:
            # sample from full exploration area
            new_exploration_area = [self.agent.min_exploration_area, self.agent.max_exploration_area]
        
        self.agent.set_exploration_area(new_exploration_area)
        return new_exploration_area
            
    def load_snapshot(self):
        try:
            snapshot = self.root_dir / "last_snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
        except:
            snapshot = self.root_dir / "second_last_snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
        for k, v in payload.items():
            setattr(self, k, v)
            if k == "wandb_run_id":
                assert wandb.run is None
                cfg = self.cfg
                exp_name = "_".join(
                    [
                        "Pretrain",
                        cfg.agent.name,
                        cfg.env.name,
                        cfg.task,
                    ]
                )
                wandb.init(
                    project=cfg.project_name,
                    group=cfg.agent.name,
                    name=exp_name,
                    id=v,
                    resume=True,
                )

    def get_snapshot_dir(self):
        if self.cfg.agent.name == "dreamer" and self.cfg.domain == "none":
            snap_dir = self.cfg.snapshot_dir.replace(
                "/none/", f"/none/{self.cfg.task}/"
            )
        else:
            snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir
        return snapshot_dir
    
def toolkit_main(cfg, maindir, workdir):
    from dreamer_pretrain import Workspace as W
    root_dir = Path.cwd()
    cfg.use_tb = False
    maindir="/mnt/home/focus" # get_original_cwd() does not work in this contenxt 

    workspace = W(cfg, maindir, workdir)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    cfg.project_name = "_".join([cfg.agent.name, cfg.domain])
    
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    workspace.train()

@hydra.main(config_path="configs", config_name="dreamer_pretrain")
def main(cfg):
    from dreamer_pretrain import Workspace as W
    root_dir = Path.cwd()

    workspace = W(cfg, maindir=get_original_cwd())
    workspace.root_dir = root_dir
    print("ROOT DIR: ", root_dir)
    snapshot = workspace.root_dir / "last_snapshot.pt"
    cfg.project_name = "_".join([cfg.project_name, cfg.domain])
    
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
        print("STARTING TRAINING!")
        workspace.train()

if __name__ == "__main__":
    main()
