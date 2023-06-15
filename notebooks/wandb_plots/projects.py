projects = {
    "RSS_explore" : ["ferrarostefano", "RSS_benchmark", "explore"], 
    "RSS_finetune" : ["ferrarostefano", "RSS_benchmark", "finetune"], 
    "RSS_task" : ["ferrarostefano", "RSS_benchmark", "task"], 
}

from collections import defaultdict
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os

expl_metrics = ['contact', 'pos_displacement', 'ang_displacement', 'vertical_displacement',
               'up_placement', 'far_placement', 'close_placement', 'left_placement', 'right_placement', 'episode_reward', 'success']

expl_metrics_keys = [ 'train/' + k for k in expl_metrics]

eval_keys = ['eval/episode_reward', 'eval/success']

config_keys = ['task', 'comment', 'seed']
    
all_keys = ['agent'] + eval_keys + config_keys + expl_metrics_keys # 1 for agent and 1 for eval_reward
    
def load_and_save(runs, project_key, save_format='numpy', mode='explore', only_finished=True):
    new_runs = []
    runs_collected = 0
    if only_finished:
        for run in runs:
            if run.state == 'finished':
                new_runs.append(run)
    else:
        new_runs = runs

    task2domain = lambda x: x.split('_')[0]

    summary = defaultdict(list)
    loaded = len(summary['task'])

    for run in tqdm(new_runs): 
        if run.id in ['w9hbqqk5', 'smegrxzp']:
            continue
        
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        if save_format == 'numpy':
            try:
                # Remove any unfinished
                if os.path.isfile(f'projects_numpy/{project_key}/{run.id}_unfinished.npy'):
                    os.remove(f'projects_numpy/{project_key}/{run.id}_unfinished.npy')
                              
                load_keys = list(np.load(f'projects_numpy/{project_key}/{run.id}.npy', allow_pickle=True).item().keys())
                if len(load_keys) == len(all_keys):
#                     print("Found")
                    runs_collected += 1
                    continue
            except FileNotFoundError:
                pass

        rw_sup = json.loads(run.config['_content']['reward_coeff'].replace("'", '"').lower())['rw_sup']

        #         print(rw_sup)
        if mode == 'explore':
            if (rw_sup == 1.0) or ('Finetune' in run.name):
                continue
        elif mode == 'task':
            if rw_sup == 0.0:
                continue
        elif mode == 'finetune':
            if (rw_sup == 1.0) or ('Finetune' not in run.name):
                continue
        else:
            raise NotImplementError()
        
        runs_collected += 1
        
        if '_content' not in run.config:
            print(run.id, 'has no _content')
            continue
        
        agent_name = json.loads(run.config['_content']['agent'].replace("'", '"').lower())['name']
        
        # alternatively to scan_history (very slow for large data) one can use history(samples=100000) to get all data
        
        train_keys = expl_metrics_keys
        train_history = list(run.scan_history(keys=train_keys))

        # Taking sum of train (exploration) metrics
        train_dict = {}
        for k in train_keys:
            train_dict[k] = [row[k] for row in train_history]
        
        eval_history = list(run.scan_history(keys=eval_keys))

        # Taking sum of eval task metrics
        eval_dict = {}
        for k in eval_keys:
            eval_dict[k] = [row[k] for row in eval_history]
            
        if save_format=='csv':
            summary['agent'].append(agent_name)
            for k in config_keys:
                summary[k].append(run.config['_content'][k])
            for k in train_keys:
                summary[k].append(sum(train_dict[k]))
            for k in eval_keys:
                summary[k].append(eval_dict[k][-1])
        elif save_format == 'numpy':
            archive = {}
            archive['agent'] = agent_name
            for k in config_keys:
                archive[k] = run.config['_content'][k]
            for k in train_keys:
                archive[k] = np.array(train_dict[k])
            for k in eval_keys:
                archive[k] = np.array(eval_dict[k])
            os.makedirs(f'projects_numpy/{project_key}', exist_ok=True)
            if only_finished:
                np.save(f'projects_numpy/{project_key}/{run.id}.npy', archive)
            else:
                np.save(f'projects_numpy/{project_key}/{run.id}_unfinished.npy', archive)
        
# TODO: rather than this better use npy files to save data, though they may be less handy to aggregate results
#       To save multiple values in a .csv
#         for i in range(len(eval_ret)):
#             summary[f'eval_ret_{i}'].append(eval_ret[i])
    
    if save_format == 'csv':
        runs_df = pd.DataFrame(summary)
        runs_df.to_csv(f'projects_csv/{project_key}.csv')
    print("Runs collected", runs_collected)