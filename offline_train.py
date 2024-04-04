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
import random

from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ, PRIMAL_TASKS 
from env.make import make
from env.utils import obs_specs

import utils
from logger import Logger
from replay_buffer import ReplayBuffer, make_replay_loader
from online_train import Workspace
from typing import Union

# Offline implementation loads a dataset of trajectories and trains the world model and the agent on it, without the need for interaction with the environment and need for any exploration
class OfflineWorkspace(Workspace):
    def __init__(self, cfg, maindir=None, workdir=None):
        super().__init__(cfg, maindir=None, workdir=None)
        
        # query the desired data for training
        cfg.dataset_dir = f"/mnt/home/datasets/{cfg.task}"
        cfg.dataset_dir = cfg.dataset_dir + "/visual_target" if cfg.vis_target_dataset else cfg.dataset_dir
        cfg.dataset_dir = cfg.dataset_dir + f"/{cfg.expl_dataset}"
        
        # reset to obtain observation specs
        self.eval_env.reset()

        # create data storage and load data
        self.replay_storage = ReplayBuffer(
            self.data_specs,
            self.meta_specs,
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

    def train(self):
        # Initialization
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, self.cfg.action_repeat)
        
        utils.init_metrics_counters(self)
        
        warmnup_profiler = 2490
        profiler_active_for = 10
        
        # clean approch for having the posibility to choose if profiling or not the algorithm
        with utils.Profiler(warmnup_profiler, profiler_active_for) if self.cfg.profile else utils.DummyProfiler() as prof:
            while train_until_step(self.global_step):
                prof.step(self.global_step)

                # save last model
                if self.global_step % 5000 == 0:
                    self.save_last_model()
                
                # save snapshot
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

                # set target position for rewarding
                if self.cfg.agent.train_target_reach:
                    if self.cfg.env.env_target:
                        target = target_pos = self.eval_env.get_target()
                    else:                        
                        if self.cfg.env.target_from_replay_bf:  # Sampling goals using batch length of 1 
                            self.replay_storage._length = 1
                            target_obs = next(self.replay_iter)
                            self.replay_storage._length = self.cfg.batch_length # Restoring replay buffer settings
                            if self.cfg.env.batch_sampling:
                                target_pos = target_obs["objects_pos"][:,:,0].cpu().numpy()
                            else:
                                target_obs = {k: v[0] for k,v in target_obs.items()} # batch size cannot be modified after initialization, so take only the first element
                                target_pos = target_obs["objects_pos"][0,0].cpu().numpy()
                        else:
                            shape = [self.cfg.batch_size, 1, 1] if self.cfg.env.batch_sampling else [1,1,1]
                            target_pos = utils.generate_target(self.eval_env.limits_exploration_area, self.cfg.curriculum_learning, self.global_step, self.cfg.env.target_modulator, self.cfg.env.target_sampling_generation_strategy, shape=shape)
                            if self.cfg.env.mixed_target or self.cfg.env.only_obs_target:
                                target_obs = self.eval_env.set_goal_state(target_pos)    
                                if target_obs["is_last"] == True: self.eval_env.reset()
                                target_obs = { k: torch.as_tensor(np.copy(v), device=self.cfg.device).unsqueeze(0) for k, v in target_obs.items()}
                    
                    if (self.cfg.env.mixed_target and random.choice([True, False])) or self.cfg.env.only_obs_target: # 50% of the time the target is the position unless specified
                        target = target_obs
                    else:
                        target = target_pos
                            
                    if self.cfg.env.mixed_target or self.cfg.env.only_obs_target:
                        self.eval_env.set_target(target_pos)  # visually set the target  
                    self.agent.set_target(target)                                  
                 
                # here i set a target but do not match the observation with the target
                obs = next(self.replay_iter) 
                if self.cfg.env.dist_as_rw: 
                    if self.cfg.task in ["reacher_easy", "reacher_hard"]:
                        obs["reward"] = - torch.sqrt(torch.sum(obs["proprio"][:,:,2:4]**2, dim=-1)).unsqueeze(-1)
                    elif self.cfg.task in ["CustomLift", "shelf-place", "bin-picking"]:
                        obs["reward"] = - torch.linalg.norm(obs["proprio"][:,:,-3:], dim=-1).unsqueeze(-1)
                    else:
                        raise ValueError("Task not implemented")
                    
                self.metrics = self.agent.update(
                    obs, self.global_step, which_policy='task'
                )[1]
                
                if should_log_scalars(self.global_step):
                    self.logger.log_metrics(self.metrics, self.global_frame, ty="train")
                    elapsed_time, _ = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                        log('fps', self.cfg.log_every_frames / elapsed_time)
                        log('step', self.global_step)
                
                if self.global_step > 0 and should_log_recon(self.global_step):
                    videos, text = self.agent.report(next(self.replay_iter))
                    self.logger.log_video(videos, self.global_frame)
                    self.logger.log_metrics(text, self.global_frame, ty="train")

                self._global_step += 1

    @utils.retry
    def save_snapshot(self):
        # divide for the different ablation experimented with (expl_dataset + vis_target + coordconv) 
        snapshot_dir = self.get_snapshot_dir() / f"expl_{self.cfg.expl_dataset}"
        if self.cfg.vis_target_dataset: snapshot_dir = snapshot_dir  / f"vis_target"
        if self.cfg.agent.world_model.encoder.coordConv: snapshot_dir = snapshot_dir  / f"coordConv" 
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
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
        return snapshot_dir
    
def toolkit_main(cfg, maindir, workdir):
    from offline_train import OfflineWorkspace as W
    root_dir = Path.cwd()
    cfg.use_tb = False
    maindir="/mnt/home/focus" # get_original_cwd() does not work in this contenxt 
    workspace = W(cfg, maindir, workdir)
    
    workspace.root_dir = root_dir
    print("ROOT DIR: ", root_dir)
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    cfg.project_name = "_".join(["offline", cfg.agent.name, cfg.domain])
    
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
        
    print("STARTING TRAINING!")
    workspace.train()

@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    from offline_train import OfflineWorkspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, maindir=get_original_cwd())

    workspace.root_dir = root_dir
    print("ROOT DIR: ", root_dir)
    snapshot = workspace.root_dir / "last_snapshot.pt"
    cfg.project_name = "_".join(["offline", cfg.project_name, cfg.domain])
    
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
