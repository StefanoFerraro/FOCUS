import os
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import warnings
warnings.filterwarnings("ignore")

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

model_free_models = ["iql", "td3", "td3_bc", "offline_td3"]

class Workspace:
    def __init__(self, cfg, maindir=None, workdir=None):
        
        self.maindir = Path.cwd() if maindir is None else maindir
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f"workspace: {self.workdir}")
        
        # setup and configaration adjustments 
        self.cfg = cfg
        domain = cfg.domain
        task = (
            cfg.task if cfg.task != "none" else PRIMAL_TASKS[domain]
        )  # -> which is the URLB default
        objets_list = globals()[domain.upper() + "_TASKS_OBJ"][task]
        self.device = torch.device(cfg.device)
        
        if "world_model" in cfg.agent.keys():
            if cfg.agent.world_model.name == "focus":
                cfg.agent.world_model.objects = objets_list
                cfg.agent.world_model.device = cfg.device
                if cfg.agent.world_model.rssm.discrete == "None": cfg.agent.world_model.rssm.discrete = None 
                
        # seed everything for reproducibility        
        utils.set_seed_everywhere(cfg.seed)

        self.logger = Logger(self.workdir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        
        # change to original working directory for loading URDF models
        os.chdir(self.maindir)  
                
        # create envs
        self.train_env = make(
            domain,
            task,
            cfg.action_repeat,
            cfg.seed,
            cfg.env,
        )

        self.eval_env = make(
            domain,
            task,
            cfg.action_repeat,
            cfg.seed,
            cfg.env,
        )

        # after env creation, change back to main directory
        os.chdir(self.maindir)

        # reset to obtain observation specs
        self.train_env.reset()
        
        # create agent
        train_obs_spec = self.train_env.obs_space
        self.agent = utils.make_dreamer_agent(
            train_obs_spec,
            self.train_env.action_spec(),
            cfg,
        )

        # get meta specs
        self.meta_specs = self.agent.get_meta_specs()

        self.data_specs = (
            *obs_specs(train_obs_spec),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBuffer(
            self.data_specs,
            self.meta_specs,
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
        
        # Initialization
        step = episode = total_reward = total_success = 0
        step_to_success = self._horizon
        step_to_success_list = []      
          
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
        video = np.empty([1, int(self._horizon/2) + 1 , 3, *self.cfg.env.renderer.size])
        
        while eval_until_episode(episode):
            episode_data = []
            
            if self.train_target_reach:     
                # set target before reset of env
                # pick random goal for evaluation
                self.eval_env.reset()
                target_tuple = self.eval_env.get_random_goal()
                target_obs = self.eval_env.set_goal_state(target_tuple[0])   
                
                if self.cfg.evaluation_target_obs:
                    target = target_obs  
                    target_pos = target_obs["objects_pos"][0]
                    target = {k: torch.as_tensor(np.copy(v), device=self.cfg.device).unsqueeze(0).unsqueeze(0) for k, v in target.items()} # add batch size and length dimensions
                else:
                    target = target_pos = target_tuple[1]
                
                self.agent.set_target(target) 
                self.eval_env.set_target(target_pos)                    
    
            obs = self.eval_env.reset()
            
            if self.cfg.vis_target_dataset and self.cfg.env.name == "rs":
                obs["rgb"] = self.eval_env.get_rgb_with_target()  
 
            if self.train_target_reach:
                # dmc envs adapt objects position after the res
                self.eval_env.set_target(target_pos)

                # double for visualization purposes
                obs["eval_rgb"] = np.concatenate([target_obs["rgb"], obs['rgb']], axis=1)
            
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

                if self.train_target_reach:
    
                    if self.cfg.vis_target_dataset and self.cfg.env.name == "rs":
                        obs["rgb"] = self.eval_env.get_rgb_with_target()  
                    
                    # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
                    if not self.cfg.env.target_ablation_diam:
                        obs["eval_rgb"] = self.eval_env.get_rgb_with_target(target_pos)
                        obs["eval_rgb"] = np.concatenate([target_obs["rgb"], obs['eval_rgb']], axis=1)
                    else:
                        obs["eval_rgb"] = obs["rgb"]
                        obs["eval_rgb"] = np.concatenate([target_obs["rgb"], obs['eval_rgb']], axis=1)

                episode_data.append(obs)
                total_reward += obs["reward"]
                step += 1
                
                # Hacky way to say that step_to_success was set
                if step_to_success == self._horizon and obs["success"]:
                    step_to_success = (step * self.cfg.action_repeat) - (episode * self._horizon)

                obj_pos = np.concatenate((obj_pos, [obs["objects_pos"][0]]))

            # log moving average, move to target metrics
            if self.train_target_reach:                  
            
                if episode == 0:
                    episode_metrics = utils.move_to_target_metrics(obj_pos, target_pos)
                    move_to_target_metrics = {k: v / self.cfg.num_eval_episodes for k, v in episode_metrics.items()}
                else:
                    episode_metrics = utils.move_to_target_metrics(obj_pos, target_pos)
                    move_to_target_metrics = {k: v / self.cfg.num_eval_episodes + move_to_target_metrics[k] for k, v in episode_metrics.items()}

                # video output for visualization   
                if episode==0:
                    video = np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)    
                else:
                    video = np.concatenate([video, np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)], axis=-1)    
                
            episode += 1
            step_to_success_list += [step_to_success]
            step_to_success = self._horizon
            total_success += obs["success"]
            obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
          
        # logging 
        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("avg_success", total_success / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("avg_step_to_success", sum(step_to_success_list) / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if self.train_target_reach:                  
                utils.log_metrics_dict(move_to_target_metrics, log)

        if self.train_target_reach:
            # B, T, C, H, W = video.shape
            video = np.uint8(video * 255)
            self.logger.log_video({' ' : video }, self.global_frame)
            
        if self.global_frame % 25000 == 0: # in order to reduce space loggin space in wandb (takes 5MB each video/TSNE)
            # Eval episodes for testing the prior
            if self.agent.name not in model_free_models and self.cfg.TSNE_analysis: utils.TSNE_analysis(self)     
            
        self.eval_env.reset()          
    
    def train(self):
        # Initialization
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat, self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, self.cfg.action_repeat)

        utils.init_metrics_counters(self)
        
        meta = self.agent.init_meta()
        agent_state = None
        obs = data = self.reset(self.train_env)            
        self.replay_storage.add(data, meta)
        self.train_target_reach = "train_target_reach" in self.cfg.agent.keys() and self.cfg.agent.train_target_reach  
        # set target position for rewarding
        if self.train_target_reach:    
            if self.cfg.env_target:
                target = self.train_env.get_target()
            else:
                target = utils.generate_target(self.train_env.limits_exploration_area, self.cfg.curriculum_learning, self.global_step, self.cfg.target_modulator)
                self.train_env.set_target(target)  # visually set the target  
                self.agent.set_target(target)  # visually set the target   

        while train_until_step(self.global_step):

            # End of episode routine
            if bool(obs["is_last"]):
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if self.metrics is not None and self.episode_step > 1:
                    
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = self.episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", self.episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                        log("success", obs["success"])
                        log("step_to_success", self.step_to_success)
                        log("contact", float(self.contact_count / episode_frame))
                        # sanity check that segmentation mask is consistent
                        log("segmentation_obj_pixels", float(self.segmentation_obj_pixels / self.episode_step))  
                        
                        obj_metrics = utils.object_metrics(self.in_areas, self.cumm_pos_displacement, self.cumm_ang_displacement, self.cumm_vertical_displacement, episode_frame)
                        utils.log_metrics_dict(obj_metrics, log)

                        if self.train_target_reach:
                            move_to_target_metrics = utils.move_to_target_metrics(self.obj_pos, target)
                            utils.log_metrics_dict(move_to_target_metrics, log)
                                
                utils.init_metrics_counters(self)
                
                # save model every 5000 steps for backup 
                if self.global_step % 5000 == 0:
                    self.save_last_model()

                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                    
                # reset env and add first observation to replay buffer
                obs = data = self.reset(self.train_env)
                self.replay_storage.add(data, meta)
                
                # reset agent state and meta informations
                meta = self.agent.init_meta()       
                agent_state = None
                
                # set target position for rewarding
                if self.train_target_reach:
                    if self.cfg.env_target:
                        target = self.train_env.get_target()
                    else:
                        target = utils.generate_target(self.train_env.limits_exploration_area, self.cfg.curriculum_learning, self.global_step, self.cfg.target_modulator)
                        self.train_env.set_target(target)  # visually set the target  
                        self.agent.set_target(target)  # condition the agent on the current target                                  
                    
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time",
                    self.timer.total_time(),
                    self.global_frame,
                )
                self.eval()
            
            meta = self.agent.update_meta(meta, self.global_step, obs)

            # take action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if seed_until_step(self.global_step):
                    action = self.train_env.act_space["action"].sample()
                else:
                    action, agent_state = self.agent.act(
                        obs,
                        meta,
                        self.global_step,
                        eval_mode=False,
                        state=agent_state,
                        finetune_mode=self.cfg.finetune
                    )
            obs = data = self.train_env.step(action)
            self.replay_storage.add(data, meta)
            
            # try to update the agent
            if not seed_until_step(self.global_step): # fill the replay buffer before training WM and agent
                if self.replay_storage._total_steps > 0:
                    if should_train_step(self.global_step):
                        self.metrics = self.agent.update(
                            next(self.replay_iter), self.global_step, which_policy="both"
                        )[1]
                    if self.metrics and should_log_scalars(self.global_step):
                        self.logger.log_metrics(self.metrics, self.global_frame, ty="train")
                    
                    if hasattr(self.agent, 'report') and callable(self.agent.report):
                        if self.global_step > 0 and should_log_recon(self.global_step):
                            videos, report_metrics = self.agent.report(next(self.replay_iter))

                            self.logger.log_video(videos, self.global_frame)
                            self.logger.log_metrics(report_metrics, self.global_frame, ty="train")

            # update counter metrics 
            utils.update_metrics_counters(self, obs)
            
    @utils.retry
    def save_snapshot(self):
        # divide for the different ablation experimented with (expl_dataset + vis_target + coordconv) 
        snapshot_dir = self.get_snapshot_dir() / f"expl_{self.cfg.expl_dataset}"
        if self.cfg.vis_target_dataset: snapshot_dir = snapshot_dir  / f"vis_target"
        if "world_model" in self.cfg.agent.keys() and self.cfg.agent.world_model.encoder.coordConv: snapshot_dir = snapshot_dir  / f"coordConv" 
        if self.cfg.agent.name == "lexa": snapshot_dir = snapshot_dir  / f"{self.cfg.agent.distance_mode}" 
        if self.cfg.env.target_ablation_diam: snapshot_dir = snapshot_dir  / f"target_ablation_diam_{int(self.cfg.env.target_ablation_diam)}"
        
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = "_".join(
            [   "Train",
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
                        "Train",
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
    from online_train import Workspace as W
    root_dir = Path.cwd()
    cfg.use_tb = False
    maindir="/mnt/home/focus" # get_original_cwd() does not work in this contenxt 
    workspace = W(cfg, maindir, workdir)
    
    workspace.root_dir = root_dir
    print("ROOT DIR: ", root_dir)
    snapshot = workspace.root_dir / "last_snapshot.pt"
    cfg.project_name = "_".join(["online", cfg.agent.name, cfg.domain])
    
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    
    print("STARTING TRAINING!")
    workspace.train()
    
@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    from online_train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, maindir=get_original_cwd())
    
    workspace.root_dir = root_dir
    print("ROOT DIR: ", root_dir)
    snapshot = workspace.root_dir / "last_snapshot.pt"
    cfg.project_name = "_".join(["online", cfg.project_name, cfg.domain])
    
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
        
    print("STARTING ONLINE TRAINING!")
    workspace.train()

if __name__ == "__main__":
    main()
