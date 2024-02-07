import warnings
warnings.filterwarnings("ignore")

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

from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ, PRIMAL_TASKS 
from env.make import make
from env.utils import obs_specs

import utils
from logger import Logger
from replay_buffer import ReplayBuffer, make_replay_loader

# Offline implementation loads a dataset of trajectories and trains the world model and the agent on it, without the need for interaction with the environment and need for any exploration
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

        self.eval_env = make(
            domain,
            task,
            cfg.obs_type,
            frame_stack,
            cfg.action_repeat,
            cfg.seed,
            cfg.env,
        )

        train_obs_spec = self.eval_env.obs_space

        os.chdir(self.maindir)

        objets_list = globals()[domain.upper() + "_TASKS_OBJ"][task]

        if cfg.agent.world_model.name == "focus":
            cfg.agent.world_model.objects = objets_list

        self.eval_env.reset()
        
        # create agent
        self.agent = utils.make_dreamer_agent(
            train_obs_spec,
            self.eval_env.action_spec(),
            cfg,
            cfg.agent,
        )

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        print(f'replay dir : {cfg.dataset_dir}')
        
        data_specs = (
            *obs_specs(train_obs_spec),
            self.eval_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBuffer(
            data_specs,
            meta_specs,
            Path(cfg.dataset_dir),
            length=cfg.batch_length,
            **cfg.replay,
            device=cfg.device,
            load_first=True
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

        step, episode, total_reward, total_success = 0, 0, 0, 0
        step_to_success = self._horizon
        step_to_success_list = []        

        # set to True to use task_behaviour, zero_shot performance
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
        video = np.empty([1, int(self._horizon/2) + 1 , 3, *self.cfg.env.renderer.size])
        move_to_target_final, move_to_target_min, move_to_target_max, move_to_target_mean = [], [], [], []
        
        while eval_until_episode(episode):
            episode_data = []
            
            # set target before reset of env
            if self.cfg.agent.train_target_reach:
                # pick goal for evaluation
                target = self.eval_env.get_random_goal()[1]
                self.agent.set_target(target) # update pos target only along x axis 
                if self.cfg.env.visualize_target:   
                    self.eval_env.set_target(target)                    
    
            obs = self.eval_env.reset()
            # dmc envs adapt objects position after the res
            if self.cfg.agent.train_target_reach and self.cfg.env.visualize_target:   
                self.eval_env.set_target(target)
                   
            obs["eval_rgb"] = obs["rgb"]
            
            episode_data.append(obs)
            agent_state = None
            while not bool(obs["is_last"]):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(
                        obs,
                        meta,
                        self.global_step,
                        eval_mode=True,
                        state=agent_state,
                    )
                obs = self.eval_env.step(action)
                
                # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
                
                if self.cfg.agent.train_target_reach and self.cfg.env.visualize_target:
                    obs["eval_rgb"] = self.eval_env.get_rgb_with_target(target)

                episode_data.append(obs)
                total_reward += obs["reward"]
                step += 1
                # Hacky way to say that step_to_success was set
                if step_to_success == self._horizon and obs["success"]:
                    step_to_success = (step * self.cfg.action_repeat) - (
                        episode * self._horizon
                    )
                # if self.agent.name == "dreamer":
                #     obj_pos = np.concatenate((obj_pos, [obs["objects_pos"]]))
                # else:
                obj_pos = np.concatenate((obj_pos, [obs["objects_pos"][0]]))

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
            total_success += obs["success"]
            obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
        
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

        # Eval episodes for testing the prior
        utils.TSNE_analysis(self)
          
    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        should_log_scalars = utils.Every(
            self.cfg.log_every_frames, self.cfg.action_repeat
        )
        should_log_recon = utils.Every(
            self.cfg.recon_every_frames, self.cfg.action_repeat
        )
        
        # Metrics init            
        metrics = None
        while train_until_step(self.global_step):
            if metrics is None:
                # log stats
                elapsed_time, total_time = self.timer.reset()

                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time",
                    self.timer.total_time(),
                    self.global_frame,
                )
                self.eval()

            if self.cfg.agent.train_target_reach:
                self.expl_area_update(self.cfg.env.target_modulator, self.cfg.curriculum_learning) # update pos target according to scheduler 
                self.agent.update_target() # update target in the agent based on the new exploration area 
                
            metrics = self.agent.update(
                next(self.replay_iter), self.global_step, which_policy='task'
            )[1]
            
            if should_log_scalars(self.global_step):
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_frames / elapsed_time)
                    log('step', self.global_step)
                    log('episode_reward', 0.)
            
            if self.global_step > 0 and should_log_recon(self.global_step):
                videos, text = self.agent.report(next(self.replay_iter))
                self.logger.log_video(videos, self.global_frame)
                self.logger.log_text(text, self.global_frame)

            self._global_step += 1

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
    from offline_train import Workspace as W
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

@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    from offline_train import Workspace as W
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
