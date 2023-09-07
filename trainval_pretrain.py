import sys
from haven import haven_wizard as hw
import hydra
import os
from dreamer_pretrain import toolkit_main

import argparse
import exp_configs
import job_configs
from pathlib import Path

@hydra.main(config_path='configs', config_name='dreamer_pretrain')
def get_config(cfg):
    global config
    config = cfg

def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    exp_dict = {k.replace('|', '.') : v for k,v in exp_dict.items() }
    sys.argv=["dreamer_pretrain.py"] + [ f"{k}={v}" for k,v in exp_dict.items()] 
    get_config()
    # original_cwd = hydra.utils.get_original_cwd()
    config.project_name = args.project_name
    # cfg.agent = Bunch(cfg.agent)
    config.snapshot_dir = f"/mnt/public/projects/{args.user}/{args.project_name}/pretrained_models/{exp_dict['agent']}/{exp_dict['env']}/{exp_dict['task']}/{exp_dict['seed']}"

    toolkit_main(config, savedir=savedir, workdir=Path.cwd())

    print("Experiment completed")

def _init_wandb(user):
    os.environ["WANDB_API_KEY"] = job_configs.WANDB_API_KEY[user]
    os.environ["WANDB_DIR"] = "/tmp/"
    os.environ["WANDB_DISABLE_CODE"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="",
        type=str,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None, type=str)
    parser.add_argument("-nw", "--num_workers", default=0, type=int)
    parser.add_argument("-u", "--user", type=str, required=True)
    parser.add_argument("-p", "--project_name", type=str, required=True)
    args, others = parser.parse_known_args()
    _init_wandb(args.user)

    if args.job_scheduler == "slurm":
        job_config = {
            "account_id": "rrg-bengioy-ad",
            "time": "12:00:00",
            "cpus-per-task": "2",
            "mem-per-cpu": "16G",
            "gres": "gpu:1",
        }
    elif args.job_scheduler == "toolkit":
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.project_name+"_pretrain"],
        job_config=job_configs.JOB_CONFIG[args.user],
        python_binary_path=job_configs.PYTHON_BINARIES[args.user],
        savedir_base=args.savedir_base,
        use_threads=True,
        args=args,
    )
