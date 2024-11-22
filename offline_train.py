import os
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import warnings
warnings.filterwarnings("ignore")

import hydra
from hydra.utils import get_original_cwd
import wandb

from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ, PRIMAL_TASKS 

import utils
from replay_buffer import ReplayBuffer, make_replay_loader
from online_train import Workspace

# Offline implementation loads a dataset of trajectories and trains the world model and the agent on it, without the need for interaction with the environment and need for any exploration
class OfflineWorkspace(Workspace):
    def __init__(self, cfg, maindir=None, workdir=None):
        super().__init__(cfg, maindir=None, workdir=None)
        
        # query the desired data directories for training
        cfg.dataset_dir = f"{cfg.dataset_dir}/{cfg.task}"
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
            # load_first=True
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
        self.train_target_reach = "train_target_reach" in self.cfg.agent.keys() and self.cfg.agent.train_target_reach  
        
        # clean approch for having the posibility to choose if profiling or not the algorithm
        while train_until_step(self.global_step):

            # save model every 5000 steps for backup 
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

            # set target position/observation to agent
            if self.train_target_reach:    
                target, _ = utils.get_target(self.cfg, self.eval_env, self.replay_storage, self.replay_iter, self.global_step)
                self.agent.set_target(target)                                  
                
            obs = next(self.replay_iter) 
                        
            if self.cfg.env.dist_as_rw: # to adapt reward signal in case of replay buffer collected with different reward 
                utils.dist_as_reward(obs, self.cfg.task)

            # agent udpate
            self.metrics = self.agent.update(
                obs, self.global_step, which_policy='task'
            )[1]
            
            # logging
            if should_log_scalars(self.global_step):
                self.logger.log_metrics(self.metrics, self.global_frame, ty="train")
                elapsed_time, _ = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_frames / elapsed_time)
                    log('step', self.global_step)
            
            if hasattr(self.agent, 'report') and callable(self.agent.report):
                if self.global_step > 0 and should_log_recon(self.global_step):
                    videos, text = self.agent.report(obs)
                    self.logger.log_video(videos, self.global_frame)
                    self.logger.log_metrics(text, self.global_frame, ty="train")

            self._global_step += 1

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
    
    print("STARTING OFFLINE TRAINING!")
    workspace.train()

if __name__ == "__main__":
    main()
