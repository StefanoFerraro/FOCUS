import itertools
import os
import sys
from tqdm import tqdm
from IPython.display import Video
from pathlib import Path
import imageio

import torch
import numpy as np

# sys.path.append("/mnt/home/focus")
# os.chdir("/mnt/home/focus")

import env
from env.make import make
import utils

import matplotlib.pyplot as plt

from evals.eval_utils import Grid, agents_loading, get_target_observation, inverse_kinematic_double_pendulum, task2domain, task2domainbench


undscr = lambda x: "_" + x if x is not None else ""

def eval(exp_dict, savedir):
    task = exp_dict["task"]
    expl_dataset = exp_dict["expl_dataset"]
    snapshot_num = exp_dict["snapshot_num"]
    vis_target = exp_dict["vis_target"]
    benchmark_id = exp_dict["benchmark_id"] # 1, 2, 3, 4_cosine, 4_temporal, 5
    benchmark = f"offline_{task2domainbench[task]}_benchmark_{benchmark_id}" + undscr(exp_dict["agent"])
    agent_name = exp_dict["agent"]
    mf_agents = ["iql", "td3_bc"]
    models = agents_loading(task, expl_dataset, snapshot_num, vis_target)
    
    # load the environment
    action_repeat = 2
    models[benchmark][0].cfg.env.target_ablation_diam = False
    eval_env = make(task2domain[task], task, action_repeat=action_repeat, seed=0, env_config=models[benchmark][0].cfg.env)
    
    # Initialization
    step = total_reward = total_success = global_step = 0
    num_pt = exp_dict["num_points"]
    
    limits_expl_area = eval_env.limits_exploration_area
    points = np.linspace(limits_expl_area[0][:2], limits_expl_area[1][:2], num=num_pt, axis=1) # in case of 3D environment testing is on the 2D place

    # concatenate over all possible combinations of points, to create a grid of points
    eval_poses = []
    for t in itertools.product(*points):
        eval_poses.append(list(t))

    # evaluate the target poses for validity
    eval_poses = np.array(eval_poses)
    valid_targets = Grid(num_pt)
    
    for i, pose in enumerate(eval_poses):
        x = i // num_pt
        y = i % num_pt
        
        # in case of reacher the target pose of the arm needs to be set since the object is the eef
        if "reacher" in task: 
            target = (inverse_kinematic_double_pendulum(pose), pose)
            if not target[0] or (list(target[1]) == [0.0, 0.0]): # in case of target not reacheable skip the episode + target pos is [0 ,0] that would null the metrics
                continue
            else:
                valid_targets[x, y] = list(target)
        else:
            pose = np.concatenate((pose, [0.05])) # in case of 3D environment testing is on the 2D place
            valid_targets[x, y] = [pose, pose]
                
    save_files = True 
    eval_env.visualize_target = vis_target
    cfg = models[benchmark][0].cfg

    valid_episodes = 0

    move_to_target_metrics = {}

    vis_target_dir = lambda s: "vis_target_" if s else ""
    final_save_dir = Path(f"{savedir}/eval_results/{task2domain[task]}/{task}/{vis_target_dir(vis_target)}/{benchmark_id}")
    if agent_name in mf_agents: final_save_dir  = final_save_dir / agent_name
    
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    for s, agent in enumerate(tqdm(models[benchmark])):
        obj_pos = np.zeros_like([cfg.env.object_start_pos]).astype(float) 
        meta = agent.init_meta
        if "video" in locals(): del video
        
        for iy, ix in tqdm(np.ndindex(valid_targets.shape)):
            episode_data = []
            eval_env.reset()
            
            # set goal from the equaly distributed set of points 
            target = valid_targets[iy, ix]
            if not target: # in case of target not reacheable skip the episode
                continue
            
            # set target before reset of env
            eval_env.set_target(target[1])                    
            if agent.name == "lexa":
                target_obs = get_target_observation(eval_env, target[0])  
                target_obs = {k: torch.as_tensor(np.copy(v), device="cuda:0").unsqueeze(0).unsqueeze(0) for k, v in target_obs.items()} # add batch size and length dimensions
                agent.set_target(target_obs) 
            elif agent.name in mf_agents:
                pass
            else:
                agent.set_target(target[1]) 
                            
            obs = eval_env.reset()
            
            # dmc envs adapt objects position after the res
            if task in ["reacher_easy", "reacher_hard"]: eval_env.set_target(target[1])                    
            
            # double for visualization purposes
            obs["eval_rgb"] = obs["rgb"]
            
            episode_data.append(obs)
            agent_state = None
            
            while not bool(obs["is_last"]):
                with torch.no_grad(), utils.eval_mode(agent):
                    action, agent_state = agent.act(
                        obs,
                        meta,
                        global_step,
                        eval_mode=True,
                        state=agent_state,
                    )
                    
                obs = eval_env.step(action)
                
                # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
                if agent.name in mf_agents or cfg.agent.train_target_reach:
                    obs["eval_rgb"] = eval_env.get_rgb_with_target()
                else:
                    obs["eval_rgb"] = obs["rgb"]

                episode_data.append(obs)
                total_reward += obs["reward"]
                step += 1
                obj_pos = np.concatenate((obj_pos, [obs["objects_pos"][0]]))

            valid_episodes += 1
            
            # log moving average, move to target metrics
            # if "train_target_reach" in cfg.agent.keys() and cfg.agent.train_target_reach:
                # target_pos = agent._target_pos.cpu().numpy()
            target_pos = target[1]
            episode_metrics = utils.move_to_target_metrics(obj_pos, target_pos)
            if not bool(move_to_target_metrics): # check if dict is empty
                move_to_target_metrics = {k: Grid(num_pt) for k in episode_metrics.keys()} # instantiate grid for each metric
            
            for k, v in episode_metrics.items():
                move_to_target_metrics[k][iy, ix] = v
                
            # video output for visualization                 
            if save_files:
                if "video" not in locals():
                    video = np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)    
                else:
                    video = np.concatenate([video, np.expand_dims(np.stack([obs['eval_rgb'] for obs in episode_data], axis=0), axis=0)], axis=-1)    
                
            total_success += obs["success"]
            obj_pos = np.zeros_like([cfg.env.object_start_pos]).astype(float) 
        
        if save_files:
            imageio.mimwrite(final_save_dir / f'{task}_{vis_target_dir(vis_target)}eval_{s}.mp4', video[0].transpose(0,2,3,1), fps=15) 
    
    if save_files:
        np.save(final_save_dir / f'{task}_{vis_target_dir(vis_target)}eval.npy', move_to_target_metrics)

if __name__ == "__main__":
    exp_dict = {"task": "Lift", "expl_dataset": "focus", "snapshot_num": 250000, "vis_target": False, "benchmark_id": "6", "num_points": 3, "agent": "td3_bc"}
    save_dir = "."
    eval(exp_dict, save_dir)
