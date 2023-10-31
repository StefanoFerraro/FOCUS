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
    "urlb_oracle_finetune" : {
        "agent": 'dreamer',
        "task": ALL_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_reward": False,
        "init_critic": False,
        "mpc": False,
        "num_train_frames": 10,
        "save_eval_episodes": True,
    },
    "urlb_lasr_finetune" : {
        "agent": "lasr",
        "task": WALKER_TASKS + QUADRUPED_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [1000000],
        "seed": [1,2,3],
        "init_reward": [True, False],
        "init_critic": False,
        "mpc": False,
    },
    "urlb_lasr_mod_finetune" : {
        "agent": "lasr",
        "task": WALKER_TASKS + QUADRUPED_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [1000000],
        "seed": [1,2],
        "init_reward": [False],
        "init_critic": False,
        "mpc": False,
    },
    "urlb_lasr_apt_pretrain" : {
        "agent": "lasr",
        "domain": ['walker','quadruped'],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_lasr_apt_long_task_pretrain" : {
        "agent": "lasr",
        "domain": ['walker','quadruped'],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_lasr_apt_finetune" : {
        "agent": "lasr",
        "task": WALKER_TASKS + QUADRUPED_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
    },
    "urlb_lasr_apt_long_task_finetune" : {
        "agent": "lasr",
        "task": WALKER_TASKS + QUADRUPED_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
    },
    "urlb_mpc_zeroshot_finetune" : {
        "agent": "plan2explore",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "num_train_frames": 10,
        "init_reward": True,
        "mpc": True,
        "mpc_opt": '{ iterations : 10, num_samples : 1000, num_elites : 100, mixture_coef : 0, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 15, use_value : False }'
    },
    "urlb_mpc_modeltrain_finetune" : {
        "agent": "plan2explore",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_reward": True,
        "mpc": True,
        "mpc_opt": '{ iterations : 10, num_samples : 1000, num_elites : 100, mixture_coef : 0, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 15, use_value : False }'
    },
    "urlb_mpc_others_finetune" : {
        "agent": ["diayn_dreamer", "rnd_dreamer"],
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_reward": [False],
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },    
    "urlb_lbs_pretrain" : {
        "agent": "lbs_dreamer",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_lbs_finetune" : {
        "agent": "lbs_dreamer",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_reward": [False],
        "mpc": False,
    }, 
    "urlb_rnd_finetune" : {
        "agent": "rnd_dreamer",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_reward": [True, False],
        "mpc": False,
    },
    "urlb_randomdreamer_finetune" : {
        "agent": "dreamer",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [0],
        "seed": [3],
        "init_reward": [False],
        "mpc": False,
    }, 
    "urlb_random_pretrain" : {
        "agent": "random_dreamer",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_freeze_decoder_finetune": {
        "agent": "plan2explore",
        "task" : ALL_TASKS,
        "obs_type": "pixels",
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_reward": [False],
        "freeze_decoder" : True,
        "save_eval_episodes": True,
        "init_policy" : True
    },
    "urlb_rnd_finetune" : {
        "agent": ["rnd_dreamer"],
        "task": JACO_TASKS,
        "obs_type": "pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,3],
        "init_reward": [False],
        "init_critic": [True],
    },    
    "urlb_random_finetune" : {
        "agent": "random_dreamer",
        "task": WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS,
        "snapshot_ts": [100000,500000,1000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_reward": [True, False],
        "mpc": False,
    },
    "urlb_expro_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_finetune" : {
        "agent": "expro",
        "task": JACO_TASKS,
        "snapshot_ts": [100000],
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_lbs_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_nce_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_p2e_nce_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_lbspro_pretrain" : {
        "agent": "lbspro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_exrec_pretrain" : {
        "agent": "exrec",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_msn_pretrain" : {
        "agent": "msn",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_msn_dad_pretrain" : {
        "agent": "msn",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_msn_learn_pretrain" : {
        "agent": "msn",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_dynamic_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_dynamic_nce_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_dynamic_approx_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_dynamic_approx_z_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_dynamic_approx_zgrad_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False
    },
    "urlb_expro_lbs_ratio1_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_ratio2_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_ratio1_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [1000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_ratio2_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_ent_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_apt_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_apt_state_pretrain" : {
        "agent": "expro",
        "domain": ALL_DOMAINS,
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_apt_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_apt_state_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_apt_half_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_lbs_half_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_apt_state_half_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2],
    },
    "urlb_expro_lbs_spc_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True
    },
    "urlb_expro_lbs_spc_horiz2_finetune" : {
        "agent": "expro",
        "task": ['jaco_reach_top_right', 'jaco_reach_bottom_right'],
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "init_policy" : False,
        "init_critic" : False
    },
    "urlb_expro_lbs_spc_freeze_decoder_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "freeze_decoder" : True
    },
    "urlb_expro_lbs_prec32_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "precision": 32
    },
    "urlb_expro_lbs_spc_gpi_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "init_policy" : False,
        "freeze_decoder" : True
    },
    "urlb_expro_lbs_spc_gpi_mean_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "init_policy" : False,
        "freeze_decoder" : True,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'

    },
    "urlb_expro_lbs_tdmpc_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "freeze_decoder" : True,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_plan_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_vote_noprior_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_vote_noprior_safe_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_vote_noprior_safe05_allplan_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_lbs_bootspc_vote_noprior_safe05_allplan_15kbuffer_finetune": {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "init_policy" : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }',
        "replay" : '{ capacity: 15e3, ongoing: False, minlen: 50, maxlen: 50, prioritize_ends: False }'
    },
    "urlb_expro_proto_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_proto_nopolicy_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
        "mpc" : True,
        "init_policy" : False,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_expro_proto_ent_nce_selector_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_lowentonly_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_skilllen8_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "imag_horizon" : 16,
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_ac_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_ac_rewnorm_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_rewnorm_noclip_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_noclip_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skilllen4_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "imag_horizon" : 16,
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skilllen8_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "imag_horizon" : 16,
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_seedrandact_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skillen1000_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_nce_selector_seperateloss_noclip_skillen100_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_expro_proto_ent_selector_seperateloss_noclip_skillen100_finetune" : {
        "agent": "expro",
        "task": ALL_TASKS,
        "snapshot_ts": [2000000],
        "obs_type": "pixels", 
        "init_policy": "False", 
        "seed": [1,2,3],
    },
    "urlb_exrec_pretrain" : {
        "agent": "exrec",
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1,2,3],
    },
    "urlb_oracle_states_pretrain" : {
        "agent": "dreamer",
        "domain": "none",
        "task": ALL_TASKS,
        "configs": "dmc_states", 
        "seed": [1,2,3],
    },

    "dmc_pixels_finetune" : {
        "agent": "dreamer",
        "task": DMC_TASKS,
        "configs": "dmc_pixels", 
        "seed": [1,2,3],
        "mpc": [True, False],
        "snapshot_ts": 0,
        "num_train_frames" : 3000010
    },
    "urlb_oracle_pixels_jaco_pretrain" : {
        "agent": "dreamer",
        "domain": "none",
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "seed": [1,2,3],
    },
    "urlb_oracle_pixels_jaco_finetune" : {
        "agent": 'dreamer',
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "num_train_frames": 10,
        "save_eval_episodes": True,
    },
    "urlb_aps_new_pretrain" : {
        "agent": "aps_dreamer",
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1,2,3],
    },
    "urlb_oracle_states_finetune" : {
        "agent": 'dreamer',
        "task": ALL_TASKS,
        "configs": "dmc_states", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "num_train_frames": 10,
        "save_eval_episodes": True,
    },
    "urlb_dreamer_exorl_offline" : {
        "agent": "dreamer",
        "configs": "dmc_states", 
        "domain": ALL_DOMAINS,
        'dataset' : 'exorl',
        'collection_method' : ['random', 'proto'],
        "seed": [1,2,3],
    },
    "urlb_dreamer_exorl_finetune" : {
    "agent": "dreamer",
    "configs": "dmc_states",
    "task": ALL_TASKS,
    "snapshot_ts": ['10000'],
    "from_offline" : True,
    'dataset' : 'exorl',
    'collection_method' : ['random', 'proto'],
    "seed": [1,2,3],
    },
    "urlb_dreamer_ours_offline" : {
        "agent": "dreamer",
        "configs": "dmc_pixels", 
        "domain": ALL_DOMAINS,
        'dataset' : 'ours',
        'collection_method' : ['random', 'lbs'],
        "seed": [1,2,3],
    },
    "urlb_exrec_exorl_offline" : {
        "agent": "exrec",
        "configs": "dmc_states", 
        "load_wm_dir": "/mnt/public/projects/sai/urlb_dreamer_exorl/offline_models",
        "snapshot_ts": 200000,
        "domain": ALL_DOMAINS,
        'dataset' : 'exorl',
        'collection_method' : ['icm', 'disagreement'],
        "seed": [1,2,3],
        "agent|knn_k" : 100 
    },
    "urlb_exrec_vqvae_top1000_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['100000','500000','1000000','2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False
    },    
    "urlb_exrec_contrastive_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['100000','500000','1000000','2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False
    },    
    "urlb_exrec_vqvae_new50_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
    },
    "urlb_exrec_vqvae_new50_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },    
    "urlb_exrec_vqvae_top50_reinforce_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_len" : [1,3,5,15,50]
    },    
    "urlb_exrec_vqvae_top50_frozen_reinforce_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|freeze_skills" : True,
        "agent|skill_len" : 1,
        "freeze_post" : True
    },    
    "urlb_exrec_vqvae_new50_p2e_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|disag_exploration" : True,
        "agent|lbs_exploration" : False,
    },
    "urlb_exrec_vqvae_new50_random_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|random_exploration" : True,
        "agent|lbs_exploration" : False,
    },
    "urlb_exrec_vqvae_new50_apt_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|apt_exploration" : True,
        "agent|lbs_exploration" : False,
    },
    'urlb_exrec_exorl_64skills_offline' : {
        'agent': 'exrec',
        'configs': 'dmc_states', 
        'domain': ALL_DOMAINS,
        'dataset' : 'exorl',
        'collection_method' : ['rnd',],
        'seed': [1,2,3],
    },
    'urlb_mb_edl_exorl_64skills_offline' : {
        'agent': 'exrec',
        'configs': 'dmc_states', 
        'domain': ALL_DOMAINS,
        'dataset' : 'exorl',
        'collection_method' : ['rnd',],
        'seed': [1,2,3],
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        "agent|skill_dim" : 64
    },
    'urlb_base_edl_exorl_64skills_coeff_offline' : {
        'agent': 'edl',
        'configs': 'dmc_states', 
        'domain': ALL_DOMAINS,
        'dataset' : 'exorl',
        'collection_method' : ['rnd',],
        'seed': [1,2,3],
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        "agent|skill_dim" : 64
    },
    'urlb_base_edl_exorl_64skills_coeff_finetune' : {
        'agent': 'edl',
        'configs': 'dmc_states',
        'task': ALL_TASKS,
        'snapshot_ts': ['200000'],
        'from_offline' : True,
        'dataset' : 'exorl',
        'collection_method' : ['rnd'],
        'seed': [1,2,3],
        "agent|skill_dim" : 64,
        'init_critic' : False,
        'init_policy' : False,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
    },
    'urlb_mb_edl_exorl_64skills_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': ALL_TASKS,
        'snapshot_ts': ['200000'],
        'from_offline' : True,
        'dataset' : 'exorl',
        'collection_method' : ['rnd'],
        'seed': [1,2,3],
        "agent|skill_dim" : 64,
        # 'init_critic' : False,
        # 'init_policy' : False,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
    },
    "urlb_exrec_states_p2e_pretrain" : {
        "agent": 'exrec',
        "domain": 'jaco',
        "configs": "dmc_states", 
        "seed": [1, 2, 3],
        "agent|disag_exploration" : True,
        "agent|lbs_exploration" : False,
    },
    "urlb_exrec_states_lbs_pretrain" : {
        "agent": 'exrec',
        "domain": 'jaco',
        "configs": "dmc_states", 
        "seed": [1, 2, 3],
    },
    "urlb_exrec_states_ar2_lbs_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_states_ar2", 
        "seed": [1, 2, 3],
    },
    "urlb_exrec_states_ar2_lbs_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_states_ar2", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },
    "urlb_aps_apt_50_pretrain" : {
        "agent": ["apt_dreamer", "aps_dreamer"],
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1,2,3],
        "agent|knn_k" : 50 
    },
    "urlb_aps_apt_50_finetune" : {
        "agent": ["apt_dreamer", "aps_dreamer"],
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },    
    "urlb_exrec_vqvae_new50_p2e_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },    
    'urlb_exrec_exorl_scratch_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': ALL_TASKS,
        'snapshot_ts': ['10000','50000', '100000', '200000'],
        'from_offline' : True,
        'dataset' : 'exorl',
        'collection_method' : ['disagreement'],
        'seed': [1,2,3],
        'init_policy': False
    },
    'urlb_exrec_states_p2e_lessent_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': JACO_TASKS,
        'snapshot_ts': ['100000', '500000','1000000', '2000000'],
        'seed': [1,2,3],
        'init_policy': False
    },
    'urlb_exrec_states_lbs_noselector_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': ALL_TASKS,
        'snapshot_ts': ['2000000'],
        'seed': [1,2,3],
        'init_policy': True,
        'agent|use_selector' : False
    },
    "urlb_exrec_vqvae_new50_random_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },    
    "urlb_exrec_vqvae_new50_apt_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
    },   
    "urlb_exrec_32skills_resampling_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 32
    }, 
    "urlb_exrec_512skills_code_resampling_ablation_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 512,
        'agent|code_resampling' : False
    },
    "urlb_exrec_mw_pretrain" : {
        "agent": 'exrec',
        "domain": 'mw',
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "mt_reward" : False
    },
    "urlb_mb_edl_mw_pretrain" : {
        "agent": 'exrec',
        "domain": 'mw',
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "mt_reward" : False,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
    },
    "urlb_base_edl_mw_pretrain" : {
        "agent": 'edl',
        "domain": 'mw',
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "mt_reward" : False,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
    },
    "urlb_mw_all_pretrain" : {
        "agent": ['aps_dreamer', 'diayn_dreamer', 'icm_dreamer', 'plan2explore', 'rnd_dreamer', 'apt_dreamer', 'lbs_dreamer'],
        "domain": 'mw',
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "mt_reward" : False
    },
    "urlb_mb_edl_mw_eval_harder_random_finetune" : {
        "agent": 'exrec',
        "task": ['mw_reach'],
        'snapshot_ts': ['2000000'],
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "init_policy": False,
        'task_id' : list(range(50)),
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True,
        'agent|update_skill_every_step' : 50
    },
    "urlb_exrec_mw_eval_harder_random_finetune" : {
        "agent": 'exrec',
        "task": ['mw_reach'],
        'snapshot_ts': ['2000000'],
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        "init_policy": False,
        'task_id' : list(range(50)),
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True,
        'agent|update_skill_every_step' : 50
    },
    "urlb_mw_all_eval_harder_random_finetune" : {
        "agent": [ 'icm_dreamer', 'plan2explore', 'rnd_dreamer', 'apt_dreamer', 'aps_dreamer', 'diayn_dreamer', 'lbs_dreamer'], 
        "task": ['mw_reach'],
        'snapshot_ts': ['2000000'],
        "configs": "mw_pixels", 
        "seed": [1, 2, 3],
        'task_id' : list(range(50)),
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True
    },
    "urlb_jaco_exrec_eval_finetune" : {
        "agent": [ 'exrec'], 
        "task": JACO_TASKS,
        'snapshot_ts': ['2000000'],
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "init_policy": False,
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True
    },
    "urlb_jaco_mb_edl_eval_finetune" : {
        "agent": [ 'exrec'], 
        "task": JACO_TASKS,
        'snapshot_ts': ['2000000'],
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "init_policy": False,
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True
    },
    "urlb_jaco_icml_all_eval_finetune" : {
        "agent": [ 'icm_dreamer', 'plan2explore', 'rnd_dreamer', 'apt_dreamer', 'aps_dreamer', 'diayn_dreamer', 'lbs_dreamer', 'random_dreamer'], 
        "task": JACO_TASKS,
        'snapshot_ts': ['2000000'],
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "num_train_frames": 10,
        "num_eval_episodes": 100,
        'eval_mw' : True
    },
    "urlb_mb_edl_64skills_resampling_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 64,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
    },
    "urlb_base_edl_64skills_pretrain" : {
        "agent": 'edl',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 64,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
    },
    "urlb_base_edl_64skills_finetune" : {
        "agent": 'edl',
        "task": ALL_TASKS,
        'snapshot_ts': ['100000', '500000', '1000000', '2000000',],
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 64,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
        'init_policy' : False,
        'init_critic' : False,
        'init_reward' : False
    },
    "urlb_mb_edl_64skills_resampling_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        'snapshot_ts': ['100000', '500000', '1000000', '2000000'],
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 64,
        'agent|edl_reward' : True,
        'agent|code_resampling' : False,
        'agent|use_selector' : False,
        'init_policy' : False,
        # 'init_critic' : False
    },
    "urlb_choreo_512skills_hyperparam_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 512,
        'agent|code_resampling' : True,
        'agent|resample_every' : [1,10,100,1000,10000],
    },
    "urlb_exrec_128skills_resampling_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 128
    },
    "urlb_exrec_512skills_resampling_pretrain" : {
        "agent": 'exrec',
        "domain": ALL_DOMAINS,
        "configs": "dmc_pixels", 
        "seed": [1, 2, 3],
        "agent|skill_dim" : 512
    },
    "urlb_aps_dynampc_finetune" : {
        "agent": ["aps_dreamer"],
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_policy": [False],
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },   
    "urlb_diayn_dynampc_finetune" : {
        "agent": ["diayn_dreamer"],
        "task": QUADRUPED_TASKS + WALKER_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": [2000000],
        "seed": [1,2,3],
        "init_policy": [True],
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },   
    'urlb_aps_new_finetune' : {
        'agent': 'aps_dreamer',
        'configs': 'dmc_pixels',
        'task': ALL_TASKS,
        'snapshot_ts': ['2000000'],
        'seed': [1,2,3],
        'init_critic': [True]
    },
    'urlb_diayn_new_finetune' : {
        'agent': 'diayn_dreamer',
        'configs': 'dmc_pixels',
        'task': ALL_TASKS,
        'snapshot_ts': ['2000000'],
        'seed': [1,2,3],
        'init_policy': [False]
    },
    "urlb_exrec_32skills_resampling_finetune" : {
        "agent": 'exrec',
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_dim" : 32
    },    
    "urlb_exrec_512skills_resampling_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['100000','500000', '1000000', '2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_dim" : 512
    },    
    "urlb_exrec_64skills_code_resampling_ablation_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['100000','500000','1000000', '2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_dim" : 64,
        'agent|code_resampling' : False,
        # 'init_critic' : False
    },    
    "urlb_exrec_512skills_code_resampling_ablation_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['100000','500000','1000000', '2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_dim" : 512,
        'agent|code_resampling' : False
    },    
    "urlb_exrec_64skills_resampling_mc_ablation_finetune" : {
        "agent": 'exrec',
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ALL_SNAPSHOTS,
        "seed": [1, 2, 3],
        "agent|use_selector" : False,
        "agent|skill_dim" : 64,
        'init_policy': False,
        'init_critic' : False
    },    
    "urlb_exrec_64skills_resampling_model_finetune" : {
        "agent": 'exrec',
        "task": ALL_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": ['2000000'],
        "seed": [1, 2, 3],
        "init_policy" : False,
        "agent|skill_dim" : 64,
        'agent|freeze_skills' : True,
        'freeze_model' : True
    },    
    'urlb_exrec_exorl_64skills_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': ALL_TASKS,
        'snapshot_ts': ['200000'],
        'from_offline' : True,
        'dataset' : 'exorl',
        'collection_method' : ['rnd'],
        'seed': [1,2,3],
        'init_policy': False,
        "agent|skill_dim" : 64,
        'init_critic' : False
    },
    'urlb_exrec_exorl_64skills_mc_ablation_finetune' : {
        'agent': 'exrec',
        'configs': 'dmc_states',
        'task': JACO_TASKS,
        'snapshot_ts': ['200000'],
        'from_offline' : True,
        'dataset' : 'exorl',
        'collection_method' : ['rnd'],
        'seed': [1,2,3],
        "agent|use_selector" : False,
        "agent|skill_dim" : 64,
        'init_policy': False,
        'init_critic' : False
    },
    'urlb_pixels_icml_all_init_critic_finetune' : {
        'agent': ['icm_dreamer', 'plan2explore', 'rnd_dreamer','lbs_dreamer','apt_dreamer','aps_dreamer','diayn_dreamer'],
        'configs': 'dmc_pixels',
        'task': ALL_TASKS,
        'snapshot_ts': [100000, 500000, 1000000],
        'seed': [1,2,3],
        'init_critic' : True
    },
    'urlb_pixels_icml_jaco_dense_radius006gaussian_finetune' : {
        'agent': ['icm_dreamer', 'plan2explore', 'rnd_dreamer','lbs_dreamer','apt_dreamer','aps_dreamer','diayn_dreamer'],
        'configs': 'dmc_pixels',
        'task': JACO_TASKS,
        'snapshot_ts': [2000000],
        'seed': [1,2,3],
        'init_critic' : False,
        'init_policy' : [False, True],
        # 'init_reward' : False
    },
    'urlb_pixels_icml_all_dynampc_finetune' : {
        'agent': ['lbs_dreamer',],
        'configs': 'dmc_pixels',
        'task': JACO_TASKS,
        'snapshot_ts': ['2000000'],
        'seed': [1,2,3],
        'init_policy' : False,
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },
    "urlb_apt_5M_finetune" : {
        "agent": ["apt_dreamer"],
        "task": JACO_TASKS,
        "configs": "dmc_pixels", 
        "snapshot_ts": [1000000, 2000000, 3000000, 4000000, 5000000],
        "seed": [1,2,3],
        "init_policy" : False, # Only Jaco
        "mpc": True,
        "mpc_opt": '{ iterations : 12, num_samples : 512, num_elites : 64, mixture_coef : 0.05, min_std : 0.1, temperature : 0.5, momentum : 0.1, horizon : 5, use_value: true }'
    },  
    
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
    
    "focus_skill_rs_pretrain" : {
        "agent": ["focus", "skill_focus"],
        "env": "rs",
        "task": ["CustomLift"],
        "seed": [1,2,3],
    },
    
    "focus_skill_dmc_pretrain" : {
        "agent": ["skill_focus"],
        "env": "dmc",
        "task": ["reacher_hard"],
        "seed": [1,2,3],
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
