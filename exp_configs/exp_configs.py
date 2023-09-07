from haven import haven_utils as hu

world_model = "focus"
agent = ['focus']
env = 'rs'
task = "CustomLift"
seed = 1
EXP_GROUPS = {

    "pretrain":  {
        "agent": agent,
        "env": env,
        "task": task, 
        "snapshot_dir": f'/mnt/public/projects/mazpie/url_benchmark/pretrained_models/{agent}/{env}/{task}/{seed}',
        "seed": seed,
    },
    
    "finetune":  {
        "agent": agent,
        "domain": env,
        "task": task,
        "snapshot_dir": f'/mnt/public/projects/mazpie/url_benchmark/pretrained_models/{agent}/{env}/{task}/{seed}',  
        "seed": seed,
    },
    }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}