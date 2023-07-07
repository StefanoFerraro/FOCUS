from haven import haven_utils as hu

world_model = "focus"
agent = ['focus']
env = 'rs'
task = "CustomLift"
seed = 1
EXP_GROUPS = {

    "pretrain":  {
        "world_model": world_model,
        "agent": agent,
        "env": env,
        "task": task, 
        "snapshot_dir": f'/mnt/public/projects/mazpie/url_benchmark/pretrained/{world_model}/{agent}/{env}/{task}/{seed}',
        "seed": seed,
        # "project_name": "FocusSkill"
    },
    
    # "finetune":  {
    #     "agent": agent_name,
    #     "domain": domain,
    #     "reward_free": False,
    #     # task settings
    #     "task": 'walker_stand',
    #     "obs_type": obs_type, # [states, pixels]
    #     "frame_stack": 3, # only works if obs_type=pixels
    #     "action_repeat": 1, # set to 2 for pixels
    #     "discount": 0.99,
    #     # train settings
    #     "num_train_frames": 2000010,
    #     "num_seed_frames": 4000,
    #     # eval
    #     "eval_every_frames": 10000,
    #     "num_eval_episodes": 10,
    #     # pretrained
    #     "snapshot_ts": 100000,
    #     "snapshot_dir": f'/mnt/colab_public/projects/mazpie/url_benchmark/pretrained/${obs_type}/${domain}/${agent_name}/${seed}',
    #     # replay buffer
    #     "replay_buffer_size": 1000000,
    #     "replay_buffer_num_workers": 4,
    #     "batch_size": 1024,
    #     "nstep": 3,
    #     "update_encoder": False, # can be either true or false depending if we want to fine-tune encoder
    #     # misc
    #     "seed": seed,
    #     "device": "cuda",
    #     "save_video": True,
    #     "save_train_video": False,
    #     "use_tb": False,
    #     "use_wandb": False,
    #     # experiment
    #     "experiment": 'exp',
    #             #"dataset": "imagenet" # the trainval.py will look into os.path.join(args.dataroot, dataset)
    # },
    }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}