from haven import haven_utils as hu
from collections import defaultdict

JACO_TASKS = ['jaco_reach_top_right', 'jaco_reach_bottom_left','jaco_reach_top_left', 'jaco_reach_bottom_right']
WALKER_TASKS = ['walker_run', 'walker_walk','walker_flip', 'walker_stand']
QUADRUPED_TASKS = ['quadruped_run', 'quadruped_walk','quadruped_jump', 'quadruped_stand']
ALL_TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS
ALL_DOMAINS = ['walker', 'jaco', 'quadruped']
ALL_SNAPSHOTS = ['100000', '500000', '1000000', '2000000']

DMC_TASKS = ['acrobot_swingup', 'cartpole_swingup_sparse', 'cheetah_run', 'finger_turn_easy', 'finger_turn_hard',
             'hopper_hop', 'quadruped_run', 'quadruped_walk', 'jaco_reach_top_left', 'reacher_easy', 'reacher_hard', 'walker_run']

AGENTS = ["dreamer", "focus", "plan2explore", "apt_dreamer"]

exp_configs = {
    "focus_rs_pretrain" : {
        "world_model": ["focus", "dreamer"],
        "agent": ["focus", "dreamer"],
        "env": "rs",
        "task": ["CustomLift", "CustomStack"],
        "seed": [1,2,3],
    },

    "focus_skill_randTarget_pretrain" : {
        "scheduler_target": False,
        "agent": "skill_focus",
        "env": "rs",
        "task": ["CustomLift", "CustomStack"],
        "seed": [1,2,3]
    },
    
    "focus_skill_rs_NormalDist_objLatent_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.9, 0.95, 0.975, 0.99, 1],
        "agent|world_model|object_encoder|mse_mode": False,
        "agent|world_model|object_extractor|mse_mode": False
    },
    
    "focus_skill_rs_MSE_objLatent_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.9, 0.95, 0.975, 0.99, 1],
        "agent|world_model|object_encoder|mse_mode": True,
        "agent|world_model|object_extractor|mse_mode": True,
    },
    
    "focus_skill_dmc_NormalDist_objLatent_kl_bias_3_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.9, 0.95, 0.975, 0.99, 1],
        "agent|world_model|object_encoder|mse_mode": False,
        "agent|world_model|object_extractor|mse_mode": False,
        "train_every_actions": [10]
    },
    
    "focus_skill_dmc_MSE_objLatent_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.95, 0.975, 1],
        "agent|world_model|object_encoder|mse_mode": True,
        "agent|world_model|object_extractor|mse_mode": True,
        "train_every_actions": [5, 10],
        "agent|world_model|object_extractor|act": ["ELU", "none"],
        "env|target_modulator": [500000, 250000]
    },
    
    "focus_skill_dmc_NormalDist_objLatent_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.95, 0.975, 1],
        "agent|world_model|object_encoder|mse_mode": False,
        "agent|world_model|object_extractor|mse_mode": False,
        "train_every_actions": [5, 10],
        "agent|world_model|object_extractor|act": ["ELU", "none"],
        "env|target_modulator": [500000, 250000]
    },
    
    "focus_skill_dmc_MSE_train_steps_sweep_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.975, 0.99],
        "agent|world_model|object_encoder|mse_mode": True,
        "agent|world_model|object_extractor|mse_mode": True,
        "train_every_actions": [5, 10]
    },
    
    "focus_skill_dmc_MSE_target_modulator_sweep_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "agent|world_model|objEnc_MSE_ratio": [0.975, 0.99],
        "agent|world_model|object_encoder|mse_mode": True,
        "agent|world_model|object_extractor|mse_mode": True,
        "env|target_modulator": [100000, 250000, 500000]
    },
    
    "focus_mw_sparse_pretrain" : {
        "agent": ["apt_dreamer", "plan2explore"],
        "env": "mw",
        "task": ["hammer", "drawer-open", "disassemble", "shelf-place", "handle-pull", "door-open", "door-close", "peg-insert-side"], 
        "use_wandb": True,
        "seed": [1, 2, 3],
    },
    
    "focus_mw_mix_rw_pretrain" : {
        "agent": ["focus"],
        "env": "mw",
        "task": ["hammer", "drawer-open", "disassemble", "shelf-place", "handle-pull", "door-open", "door-close", "peg-insert-side"], 
        "use_wandb": True,
        "seed": [1, 2, 3],
    },
    
    "focus_mw_sparse_finetune" : {
        "agent": "focus",
        "env": "mw",
        "task": ["drawer-open", "disassemble", "shelf-place", "handle-pull", "door-open", "door-close", "peg-insert-side"],
        "use_wandb": True,
        "seed": [1, 2, 3],
        "num_train_frames": [500010],
    },
    
    "dreamer_mw_sparse_pretrain" : {
        "agent": "dreamer",
        "env": "mw",
        "task": ["hammer"], # "drawer-open", "disassemble", "shelf-place", "handle-pull", "door-open", "door-close", "peg-insert-side", 
        "use_wandb": True,
        "seed": [1, 2, 3],
    },
    
    "random_dreamer_sparse_pretrain" : {
        "agent": "random_dreamer",
        "env": "ms",
        "task": "CustomLiftYCB", # "drawer-open", "disassemble", "shelf-place", "handle-pull", "door-open", "door-close", "peg-insert-side", 
        "env|objects|name": ["011_banana", "002_master_chef_can"],
        "env|task_reward": ["lift", "push"],
        "use_wandb": True,
        "seed": [1, 2, 3],
    },
    
    "skill_focus_reacher_easy_dist_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_easy"],
        "seed": [1,2,3],
        "is_finetune": True,
        "agent|world_model|object_extractor|obj_latent_as_dist": True,
        "agent|world_model|object_encoder|distance_mode": "kl",        
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints"
    },
    
    "focus_skill_dmc_sweep_distance_mode_coordConv_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2],
        "agent|world_model|objEnc_MSE_ratio": [0.95, 0.99],
        "agent|world_model|object_extractor|obj_latent_as_dist": False,
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["mse", "cosine"],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|coordConv": [False],  
        "num_train_frames": 1000010,  
        "curriculum_learning": False, 
    },
    
    "focus_skill_dmc_test_no_deter_features_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2],
        "agent|world_model|objEnc_MSE_ratio": [0.95, 0.99],
        "agent|world_model|object_extractor|obj_latent_as_dist": False,
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["mse", "cosine"],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|coordConv": [False],  
        "num_train_frames": 1000010,  
        "curriculum_learning": False, 
    },
    
    "online_rs_focus_test_pretrain" : {
        "agent": ["focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
    }, 
       
    "focus_skill_rs_test_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_extractor|obj_latent_as_dist": False,
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
    },
    
    "dreamer_reacher_hard_position_rw_pretrain" : {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "train_every_actions": [5, 10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
    },
    
    "dreamer_reacher_hard_only_pos_rw_pretrain" : {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
    },  
    
    "focus_reacher_hard_position_rw_pretrain" : {
        "agent": ["focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
        "train_every_actions": [5, 10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
    },
    
    "dreamer_reacher_only_test_states_pretrain" : {
        "agent": ["dreamer"],
        "env": ["dmc"],
        "task": ["reacher_hard"],
        "seed": [1,2],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
        "snapshots": [[10000, 25000, 50000, 100000, 150000, 200000, 500000, 1000000]],
        "agent|reward_fn": ["task"], 
        "dataset_dir": "/mnt/home/datasets/reacher_hard/plan2explore/2M"
    },  
    
    "skill_dreamer_reacher_test_1_pretrain" : {
        "agent": ["skill_dreamer"],
        "env": ["dmc"],
        "task": ["reacher_hard"],
        "seed": [1,2],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
        "agent|start_agent_training_after": 2500,
        "snapshots": [[10000, 25000, 50000, 100000, 150000, 200000, 500000, 1000000]],
        "dataset_dir": "/mnt/home/datasets/reacher_hard/plan2explore/2M"
    },  
    
    "dreamer_reacher_test_proof_pretrain" : {
        "agent": ["dreamer"],
        "env": ["rs"],
        "task": ["CustomLift"],
        "seed": [1,2],
        "train_every_actions": [10],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "curriculum_learning": False,
        "snapshots": [[10000, 25000, 50000, 100000, 150000, 200000, 500000, 1000000]],
        "agent|reward_fn": ["task"] 
    },  

    "offline_reacher_benchmark_1_test_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|reward_fn": ["task"],
        "agent|train_target_reach": [True],
        "num_train_frames": 250000,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "dataset_dir": "/mnt/public/projects/mazpie/online_reacher_benchmark_1/pretrain/861fcdbba354d1d39ed21ec4a6201d73/code/buffer"
    },
    
    "offline_reacher_online_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_hard", "reacher_easy"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|reward_fn": ["pos"],
        "agent|world_model|encoder|mlp_keys": "objects_pos",
        "agent|world_model|decoder|mlp_keys": "objects_pos",
        "agent|world_model|encoder|norm": "layer",
        "agent|world_model|decoder|norm": "layer",
        "num_train_frames": 100000,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500,
        "dataset_dir": "/mnt/home/datasets/reacher_hard/plan2explore/2M"
    },  
    
    # FOCUS++ dataset acquisition 
    
    "online_reacher_benchmark_1_dataset_no_visual_target_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "eval_every_frames": 10000000,
        "env|visualize_target": False,
    },  
    
    "online_mw_benchmark_1_dataset_pretrain": {
        "agent": ["dreamer"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "eval_every_frames": 100000,
    },  

    "online_reacher_plan2explore_dataset_no_visual_target_pretrain": {
        "agent": ["plan2explore"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "eval_every_frames": 10000000,
        "env|visualize_target": False,
    },  
    
    "online_mw_plan2explore_dataset_pretrain": {
        "agent": ["plan2explore"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "eval_every_frames": 100000,
    }, 
            
    "online_reacher_benchmark_5_dataset_no_visual_target_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],        
        "eval_every_frames": 10000000,
        "env|visualize_target": False,
    }, 
    
    "online_mw_benchmark_5_dataset_pretrain" : {
        "agent": ["skill_focus"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],        
        "eval_every_frames": 100000,
    }, 
    
    # Online Reacher Benchmarking FOCUS++
     
    "online_reacher_benchmark_1_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
    },  
    
    "offline_reacher_benchmark_1_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500,
        "vis_target_dataset": [True, False], 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },
    
    "online_reacher_benchmark_1_state_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|mlp_keys": "proprio",
        "agent|world_model|decoder|mlp_keys": "proprio",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "agent|symlog_inputs": [True],
    },  
    
    "online_reacher_benchmark_2_pretrain": {
        "agent": ["skill_dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "env|horizon": [250, 1000], 
    },  
    
    "offline_reacher_benchmark_2_pretrain": {
        "agent": ["skill_dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "vis_target_dataset": [False, True], 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
      
    "online_reacher_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|cnn_keys": "rgb",
        "agent|world_model|decoder|cnn_keys": "rgb",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "env|dist_as_rw": [True],
        "agent|world_model|encoder|coordConv": [False, True], # Ablation with CoordConv
    },  
    
    "offline_reacher_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|cnn_keys": "rgb",
        "agent|world_model|decoder|cnn_keys": "rgb",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "env|dist_as_rw": [True],
        # "agent|world_model|encoder|coordConv": [False, True], # Ablation with CoordConv
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "vis_target_dataset": [False, True], 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
    
    "offline_reacher_benchmark_4_pretrain": {
        "agent": ["lexa"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|distance_mode": ["cosine", "temporal"],
        "num_train_frames": 250010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "vis_target_dataset": [False, True], 
        "expl_dataset": ["dreamer", "p2e", "focus"],
        "env|target_from_replay_bf": True,
        "env|only_obs_target": True,
        "env|batch_sampling": [True],
        "evaluation_target_obs": True,
    },  

    "online_reacher_benchmark_5_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_easy", "reacher_hard"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "env|horizon": [250, 1000], # shorter horizon converge faster, longer horizon gets more precise 
    }, 
    
    "offline_reacher_benchmark_5_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_easy"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        # "env|horizon": [250, 1000], # shorter horizon converge faster, longer horizon gets more precise 
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "vis_target_dataset": [False], 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    }, 
    
    # Online Robosuite Benchmarking FOCUS++
    
    "online_rs_benchmark_1_pretrain": {
        "agent": ["dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
    },  
    
    "offline_rs_benchmark_1_pretrain": {
        "agent": ["dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
    
    "online_rs_benchmark_1_state_pretrain": {
        "agent": ["dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|mlp_keys": "proprio",
        "agent|world_model|decoder|mlp_keys": "proprio",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "agent|symlog_inputs": [True],
    },  
    
    "online_rs_benchmark_2_pretrain": {
        "agent": ["skill_dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "env|horizon": [250, 1000], 
    },  
    
    "offline_rs_benchmark_2_pretrain": {
        "agent": ["skill_dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
      
    "online_rs_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|cnn_keys": "rgb",
        "agent|world_model|decoder|cnn_keys": "rgb",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "env|dist_as_rw": [True],
        # "agent|world_model|encoder|coordConv": [False, True], # Ablation with CoordConv
    },  
    
    "offline_rs_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|cnn_keys": "rgb",
        "agent|world_model|decoder|cnn_keys": "rgb",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "env|dist_as_rw": [True],
        # "agent|world_model|encoder|coordConv": [False, True], # Ablation with CoordConv
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
    
    "offline_rs_benchmark_4_pretrain": {
        "agent": ["lexa"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|distance_mode": ["cosine", "temporal"],
        "num_train_frames": 1000010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
        "env|target_from_replay_bf": True,
        "env|only_obs_target": True,
        "env|batch_sampling": [True],
        "evaluation_target_obs": True,
    },  

    "online_rs_benchmark_5_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["cosine"],
    }, 
    
    "offline_rs_benchmark_5_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    }, 
    
    "offline_rs_benchmark_5_test_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
    }, 
    
    "offline_rs_benchmark_5_pos_from_rb_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 150010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "env|target_from_replay_bf": True,
        "env|mixed_target": False 
    }, 
    
    "offline_rs_benchmark_5_mixed_rendered_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2, 3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "env|target_from_replay_bf": False,
        "env|mixed_target": True 
    }, 
    
        "offline_rs_benchmark_5_only_rendered_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2, 3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["mse", "cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "env|target_from_replay_bf": False,
        "env|mixed_target": False,
        "env|only_obs_target": True
    }, 

    "offline_rs_benchmark_5_reduced_latent_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "agent|world_model|objlatent_ratio": [0.8, 0.9, 0.95, 0.98],
    },  
    
    "offline_rs_benchmark_5_sampling_ablation_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2, 3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "env|target_sampling_generation_strategy": ["uniform", "normal"],
        "env|batch_sampling": [False, True],
    }, 
    
    "offline_rs_benchmark_5_replay_buffer_pretrain" : {
        "agent": ["skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1, 2, 3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["focus"],
        "env|target_from_replay_bf": True,
        "env|batch_sampling": [False, True],
    }, 
    
     # Offline MetaWorld Benchmarking FOCUS++
    
    "offline_mw_benchmark_1_pretrain": {
        "agent": ["dreamer"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
        
    },  
    
    "offline_mw_benchmark_2_pretrain": {
        "agent": ["skill_dreamer"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
    
    "offline_mw_benchmark_3_pretrain": {
        "agent": ["dreamer"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|world_model|encoder|cnn_keys": "rgb",
        "agent|world_model|decoder|cnn_keys": "rgb",
        "agent|train_target_reach": [True],
        "agent|reward_fn": ["task"],
        "env|dist_as_rw": [True],
        # "agent|world_model|encoder|coordConv": [False, True], # Ablation with CoordConv
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    },  
    
    "offline_mw_benchmark_4_pretrain": {
        "agent": ["lexa"],
        "env": "mw",
        "task": ["shelf-place"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|distance_mode": ["cosine", "temporal"],
        "num_train_frames": 1000010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
        "env|target_from_replay_bf": True,
        "env|only_obs_target": True,
        "env|batch_sampling": [True],
        "evaluation_target_obs": True,
    },  

    "offline_mw_benchmark_5_pretrain": {
        "agent": ["skill_focus"],
        "env": "mw",
        "task": ["bin-picking"],
        "seed": [1,2,3],
        "env|segmenter|checkpoints_folder": "/mnt/home/focus/checkpoints",
        "agent|train_target_reach": [True],
        "agent|symlog_inputs": [True],
        "agent|world_model|objEnc_MSE_ratio": [0.99],
        "agent|world_model|object_encoder|distance_mode": ["cosine"],
        "agent|distance_mode": ["cosine"],
        "num_train_frames": 500010,  
        "log_every_frames": 1000,
        "recon_every_frames": 2500,
        "eval_every_frames": 2500, 
        "expl_dataset": ["dreamer", "p2e", "focus"],
    }, 
}

EXP_GROUPS = { k : hu.cartesian_exp_group(v) for k,v in exp_configs.items()} 

# Custom snap_dir is a special case
# for d in EXP_GROUPS["urlb_oracle_states_finetune"]:
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/urlb_oracle_states/pretrained_models/states/none/{d['task']}/dreamer"
# for d in EXP_GROUPS["urlb_oracle_pixels_jaco_finetune"]:
#     d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_oracle_pixels_jaco/pretrained_models/pixels/none/{d['task']}/dreamer"
# for d in EXP_GROUPS["urlb_oracle_finetune"]:
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/none/{d['task']}/dreamer"
# for d in EXP_GROUPS["urlb_rnd_finetune"]:
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/none/{d['task']}/dreamer"
# for d in EXP_GROUPS["urlb_mpc_zeroshot_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/{domain}/{d['agent']}"
# for d in EXP_GROUPS["urlb_mpc_modeltrain_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/{domain}/{d['agent']}"
# for d in EXP_GROUPS["urlb_mpc_others_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/{domain}/{d['agent']}"
# for d in EXP_GROUPS["urlb_freeze_decoder_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/sai/url_plan2explore_all/pretrained/pixels/{domain}/{d['agent']}"
#     if domain == 'jaco':
#         d["init_policy"] = False
# for d in EXP_GROUPS["urlb_expro_apt_half_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_apt/pretrained_models/pixels/{domain}/{d['agent']}/"
# for d in EXP_GROUPS["urlb_expro_apt_state_half_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_apt_state/pretrained_models/pixels/{domain}/{d['agent']}/"
# for d in EXP_GROUPS["urlb_expro_lbs_half_finetune"]:
#     domain, _ = d['task'].split('_', 1)
#     d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_lbs/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ["urlb_expro_lbs_spc_finetune", "urlb_expro_lbs_spc_horiz2_finetune","urlb_expro_lbs_spc_freeze_decoder_finetune", "urlb_expro_lbs_prec32_finetune",
#             "urlb_expro_lbs_spc_gpi_finetune", "urlb_expro_lbs_spc_gpi_mean_finetune", "urlb_expro_lbs_tdmpc_finetune", "urlb_expro_lbs_bootspc_finetune", "urlb_expro_lbs_bootspc_plan_finetune",
#             "urlb_expro_lbs_bootspc_vote_noprior_finetune", "urlb_expro_lbs_bootspc_vote_noprior_safe_finetune", "urlb_expro_lbs_bootspc_vote_noprior_safe05_allplan_finetune", "urlb_expro_lbs_bootspc_vote_noprior_safe05_allplan_15kbuffer_finetune"]:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_lbs/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ["urlb_expro_proto_nopolicy_finetune"]:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_proto/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ["urlb_expro_proto_ent_nce_selector_finetune", "urlb_expro_proto_ent_nce_selector_lowentonly_finetune", 
#             "urlb_expro_proto_ent_nce_selector_seperateloss_finetune", "urlb_expro_proto_ent_nce_selector_seperateloss_skilllen8_finetune",
#             "urlb_expro_proto_ent_nce_ac_finetune", "urlb_expro_proto_ent_nce_ac_rewnorm_finetune", "urlb_expro_proto_ent_nce_selector_seperateloss_rewnorm_noclip_finetune",
#             "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_finetune", "urlb_expro_proto_ent_nce_selector_noclip_finetune",
#             "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skilllen4_finetune", "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skilllen8_finetune",
#             "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_seedrandact_finetune", "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skillen1000_finetune",
#             "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skillen100_finetune"]:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/sai/urlb_expro_proto_ent_nce/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ["urlb_expro_proto_ent_selector_seperateloss_noclip_skillen100_finetune"]:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_expro_proto_ent/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ["urlb_exrec_vqvae_top50_reinforce_finetune"]:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_vqvae_top50/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_aps_dynampc_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_aps_new/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_diayn_dynampc_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/sai/urlb_diayn_new/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_exrec_states_lbs_noselector_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_states_lbs/pretrained_models/states/{domain}/{d['agent']}/"
# # for exp in ['urlb_exrec_exorl_scratch_reversed_rewgrad_finetune']:
# #     for d in EXP_GROUPS[exp]:
# #         domain, _ = d['task'].split('_', 1)
# #         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_exorl_scratch_reversed/offline_models/{d['dataset']}/{d['collection_method']}/{domain}/{d['agent']}/"
# for exp in ['urlb_pixels_icml_all_init_critic_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_pixels_icml_all/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_jaco_icml_all_eval_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_pixels_icml_all/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_jaco_exrec_eval_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_64skills_resampling/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_exrec_64skills_resampling_model_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_64skills_resampling/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_pixels_icml_jaco_dense_radius006gaussian_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/home/urlb_paper_backups/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_exrec_mw_eval_harder_random_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_mw/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_mb_edl_mw_eval_harder_random_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_mb_edl_mw/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_mw_all_eval_harder_random_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_mw_all/pretrained_models/pixels/{domain}/{d['agent']}/"
# # Ablations rebuttal
# for exp in ['urlb_exrec_64skills_resampling_mc_ablation_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_64skills_resampling/pretrained_models/pixels/{domain}/{d['agent']}/"
# for exp in ['urlb_exrec_exorl_64skills_mc_ablation_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_exrec_exorl_64skills/offline_models/{d['dataset']}/{d['collection_method']}/{domain}/{d['agent']}/"
# for exp in ['urlb_jaco_mb_edl_eval_finetune']:
#     for d in EXP_GROUPS[exp]:
#         domain, _ = d['task'].split('_', 1)
#         d["custom_snap_dir"] = f"/mnt/public/projects/mazpie/urlb_mb_edl_64skills_resampling/pretrained_models/pixels/{domain}/{d['agent']}/"
