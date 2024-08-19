from haven import haven_wizard as hw

import argparse
import exp_configs
import job_configs
import importlib

def eval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    config_dict = {k.replace('|', '.') : v for k,v in exp_dict.items()}
    
    savedir = f"../../../"
    module_name = f'evals.{str(config_dict["evaluation_script"])}'
    eval_module = importlib.import_module(module_name)
    eval_module.eval(exp_dict, savedir=savedir)

    print("Evaluation completed")

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

    job_config = job_configs.JOB_CONFIG[args.user]
    job_config["resources"]["gpu_model"] = "P100"
    
    hw.run_wizard(
        func=eval,
        exp_list=exp_configs.EXP_GROUPS[args.project_name+"_eval"],
        job_config=job_configs.JOB_CONFIG[args.user],
        python_binary_path=job_configs.PYTHON_BINARIES[args.user],
        savedir_base=args.savedir_base,
        use_threads=True,
        args=args,
    )
