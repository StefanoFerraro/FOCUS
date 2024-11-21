import collections
from pathlib import Path 
from collections import defaultdict
import os
import numpy as np
import torch
import math 

os.environ['MUJOCO_GL'] = 'egl'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# load relevant models
benchmark2agent = {"1": "dreamer", "2": "skill_dreamer", "3": "dreamer", "4": "lexa", "5": "skill_focus", "6": "iql", "7": "focus", "8": "focus_pcp"}
task2domain = {"reacher_easy": "dmc", "reacher_hard": "dmc", "Lift": "rs", "shelf-place": "mw", "bin-picking": "mw"}
task2domainbench = {"reacher_easy": "reacher", "reacher_hard": "reacher", "Lift": "rs", "shelf-place": "mw", "bin-picking": "mw"}

base_path = Path("")

distance_modes = ["cosine", "temporal"]
mf_agents = ["iql", "td3_bc"]
seeds = [1, 2, 3]

def load_agent(agent_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with agent_path.open('rb') as f:
        obj = torch.load(f, map_location=torch.device(device))
        agent = obj['agent']    
        step = obj['_global_step']
    return agent, step

def load(agent_path, device='cuda'):
    agent, step = load_agent(agent_path)
    # agent = init_agent(configs[task][id])
    agent.device = device
    try: # in case of model free agents
        agent.wm.device = device
        agent.wm.rssm.device = device
        agent.wm.rssm._cell.device = device
    except:
        pass
    return agent

def agents_loading(task, expl_dataset, snapshot_num, vis_target=False):
    
    expl_dataset = f"expl_{expl_dataset}"
    snapshot = f"snapshot_{str(snapshot_num)}"    
    benchmarks = [f"offline_{task2domainbench[task]}_benchmark_{i}" for i in [1,2,3,4,5,6,7,8]]
    
    models = defaultdict(list)
    
    for benchmark in benchmarks: 
        for seed in seeds:
            benchmark_id = benchmark.split("_")[-1]
            path = Path(base_path) / benchmark / "pretrained_models" / benchmark2agent[benchmark_id] / task2domain[task] / task / str(seed) / expl_dataset
            if vis_target: path = path / "vis_target"
            if benchmark_id == "4": # in case of lexa  select the desired distance mode for testing
                for dist in distance_modes:
                    path_lexa = path / dist
                    path_lexa = path_lexa / f"{snapshot}.pt"
                    models[f"{benchmark}_{dist}"].append(load(path_lexa))
                    
            elif benchmark_id == "6": # in case of lexa  select the desired distance mode for testing
                for mf_agent in mf_agents:
                    path = Path(base_path) / benchmark / "pretrained_models" / mf_agent / task2domain[task] / task / str(seed) / expl_dataset
                    path = path / f"{snapshot}.pt"
                    models[f"{benchmark}_{mf_agent}"].append(load(path))
            else:
                path = path / f"{snapshot}.pt"
                # load model from path
                models[benchmark].append(load(path))
    return models


def flatten_observation(observation, output_key='observations'):
  if not isinstance(observation, collections.abc.MutableMapping):
    raise ValueError('Can only flatten dict-like observations.')

  if isinstance(observation, collections.OrderedDict):
    keys = observation.keys()
  else:
    # Keep a consistent ordering for other mappings.
    keys = sorted(observation.keys())

  observation_arrays = [observation[key].ravel() for key in keys]
  return type(observation)([(output_key, np.concatenate(observation_arrays))])

def get_target_observation(eval_env, goal_pose):
    
    obs = eval_env.set_goal_state(goal_pose) 
    # action = np.zeros_like(eval_env.act_space["action"].sample())   
    # obs = eval_env.step(action)
    return obs

def inverse_kinematic_double_pendulum(target):
    l = 0.12
    x, y = target
    try:
        q_2 = math.acos((x**2 + y**2 - 2*l**2) / (2*l**2))
        q_1 = math.atan2(y, x) - math.atan2(l*math.sin(q_2), l + l*math.cos(q_2))
        return (q_1, q_2)
    except:
        return None # in case computation is not possible
    
class Grid:
    def __init__(self, size):
        self.grid = np.empty((size, size), dtype=object) 
        self.num_valid_values = 0
        
    def __getitem__(self, index: list):
        return self.grid[index[0]][index[1]]
    
    def __setitem__(self, index: list, value):
        if value is None:
            raise ValueError("Value cannot be None")
        
        if self.grid[index[0]][index[1]] is not None:
             self.grid[index[0]][index[1]].append(value)
        else:
            if isinstance(value, list):
                self.grid[index[0]][index[1]] = value
            else:
                self.grid[index[0]][index[1]] = [value]
        self.num_valid_values += 1
        
    def get_all_values(self):
        return np.array([x for x in self.grid.flatten() if x is not None]).flatten()
    
    def get_stats_over_grid_elements(self):
        means = np.zeros_like(self.grid)
        std = np.zeros_like(self.grid)
        
        for iy, ix in np.ndindex(self.grid.shape):
            if self.grid[iy, ix] is not None:
                means[iy, ix] = np.mean(self.grid[iy, ix])
                std[iy, ix] = np.std(self.grid[iy, ix])                
        return means, std

    @property
    def shape(self):
        return self.grid.shape