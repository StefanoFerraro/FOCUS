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
        if cfg.agent.world_model.name == "focus":
            cfg.agent.world_model.objects = objets_list
        self.device = torch.device(cfg.device)
        cfg.agent.world_model.device = cfg.device
        
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
        
        # Initialization
        step = episode = total_reward = total_success = 0
        step_to_success = self._horizon
        step_to_success_list = []        
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
        video = np.empty([1, int(self._horizon/2) + 1 , 3, *self.cfg.env.renderer.size])
        move_to_target_final = move_to_target_min =  move_to_target_max =  move_to_target_mean = []
        
        while eval_until_episode(episode):
            episode_data = []
            
            # set target before reset of env
            if self.cfg.agent.train_target_reach:
                # pick random goal for evaluation
                target = self.eval_env.get_random_goal()[1]
                self.agent.set_target(target) 
                if self.cfg.env.visualize_target: # visualize target if required   
                    self.eval_env.set_target(target)                    
    
            dreamer_obs = self.eval_env.reset()
            
            # dmc envs adapt objects position after the res
            if self.cfg.agent.train_target_reach and self.cfg.env.visualize_target:   
                self.eval_env.set_target(target)

            # double for visualization purposes
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
                    step_to_success = (step * self.cfg.action_repeat) - (episode * self._horizon)

                obj_pos = np.concatenate((obj_pos, [dreamer_obs["objects_pos"][0]]))

            # log specs
            if self.cfg.agent.train_target_reach:
                target_pos = self.agent._target_pos.cpu().numpy()
                move_to_target_final.append(np.exp(- np.linalg.norm(
                        obj_pos[-1] - target_pos) / np.linalg.norm(target_pos))) # exponential distance from the target at the end of episode
                move_to_target_min.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).max()) # exponential min distance to target during the entire episode
                move_to_target_max.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).min()) # exponential max distance to target during the entire episode
                move_to_target_mean.append(np.exp(- np.linalg.norm(
                        obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).mean()) # exponential max distance to target during the entire episode
            
            # video output for visualization                 
            if episode==0:
                video = np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)    
            else:
                video = np.concatenate([video, np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)], axis=-1)    
            
            episode += 1
            step_to_success_list += [step_to_success]
            step_to_success = self._horizon
            total_success += dreamer_obs["success"]
            obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 
          
        # logging 
        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("avg_success", total_success / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("avg_step_to_success", sum(step_to_success_list) / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if self.cfg.agent.train_target_reach:
                log("move_to_target_final", np.mean(move_to_target_final))
                log("move_to_target_min", np.mean(move_to_target_min))
                log("move_to_target_max", np.mean(move_to_target_max))
                log("move_to_target_mean", np.mean(move_to_target_mean))

        # B, T, C, H, W = video.shape
        video = np.uint8(video * 255)
        self.logger.log_video({'eval_video' : video }, self.global_frame)

        # Eval episodes for testing the prior
        utils.TSNE_analysis(self)         
          
    def train(self):
        
        # Initialization
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat, self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, self.cfg.action_repeat)

        episode_step = 0
        step_to_success = self._horizon
        episode_reward = 0
        agent_state = None
        meta = self.agent.init_meta()
        
        dreamer_obs = self.reset(self.train_env)            
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
        
        warmnup_profiler = 490
        profiler_active_for = 10
        
        # clean approch for having the posibility to choose if profiling or not the algorithm
        with utils.Profiler(warmnup_profiler, profiler_active_for) if self.cfg.profile else utils.DummyProfiler() as prof:
            while train_until_step(self.global_step):
                prof.step(self.global_step)

                if bool(dreamer_obs["is_last"]):
                    self._global_episode += 1
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
                            log("left_placement", float(in_areas[0] / episode_frame))
                            log("right_placement", float(in_areas[1] / episode_frame))
                            log("close_placement", float(in_areas[2] / episode_frame))
                            log("far_placement", float(in_areas[3] / episode_frame))
                            log("up_placement", float(in_areas[4] / episode_frame))
                            log("pos_displacement", cumm_pos_displacement)
                            log("ang_displacement", cumm_ang_displacement)
                            log("vertical_displacement", cumm_vertical_displacement)
                            if self.cfg.agent.train_target_reach:
                                target_pos = self.agent._target_pos.cpu().numpy()
                                # exponential distance from the target at the end of episode
                                log("move_to_target_final", np.exp(- np.linalg.norm( obj_pos[-1] - target_pos) / np.linalg.norm(target_pos)))
                                # exponential min distance to target during the entire episode
                                log("move_to_target_min", np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).max()) 
                                # exponential max distance to target during the entire episode
                                log("move_to_target_max", np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).min()) 
                                 # exponential max distance to target during the entire episode
                                log("move_to_target_mean", np.exp(- np.linalg.norm(obj_pos - target_pos, axis=-1) / np.linalg.norm(target_pos)).mean())
                                # episode average number of pixels for main object segmentation mask
                                log("segmentation_obj_pixels", float(segmentation_obj_pixels / episode_step))  
                            
                    contact_count = 0
                    in_areas = np.array([0, 0, 0, 0, 0])
                    obj_pos = np.zeros_like([self.cfg.env.object_start_pos]).astype(float) 

                    cumm_pos_displacement = 0
                    cumm_ang_displacement = 0
                    cumm_vertical_displacement = 0
                    segmentation_obj_pixels = 0
                    
                    # save last model
                    if self.global_step % 5000 == 0:
                        self.save_last_model()

                    # reset env
                    dreamer_obs = self.reset(self.train_env)
                    
                    # set target position for rewarding
                    if self.cfg.agent.train_target_reach:
                        self.expl_area_update(self.cfg.env.target_modulator, self.cfg.curriculum_learning) # update pos target according to scheduler 
                        self.agent.update_target() # update target in the agent based on the new exploration area 
                        self.train_env.set_target(self.agent.get_target()[0,0,0].detach().cpu().numpy().copy())  # visually set the target                  
                    
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
                                next(self.replay_iter), self.global_step, which_policy="both"
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
    from online_train import Workspace as W
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
    from online_train import Workspace as W
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
