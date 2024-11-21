from env.custom_maniskill_tasks.custom_lift import CustomLiftCubeEnv
from env.custom_maniskill_tasks.custom_stack import CustomStackCubeEnv
from env.custom_maniskill_tasks.move_to import MoveToEnv
from env.custom_maniskill_tasks.custom_liftYCB import CustomLiftYCBEnv

import gym

def make(task, env_args):
                
    kwargs = {
                "obs_mode": env_args.obs_mode,
                "control_mode": env_args.controller,
                "reward_mode": env_args.reward_mode,
                "camera_cfgs": {
                    "add_segmentation": True,
                    "height": env_args.size[0],
                    "width": env_args.size[1],
                    "texture_names": ("Color", "Position", "Segmentation"),}
            }
    
    if task == "LiftCube-v0" or task == "StackCube-v0":
        return gym.make(
            task,
            box_half_size=env_args.cube_minsize,
            cube_rgba=env_args.cube_rgba[:3], 
            spawn_range=env_args.spawn_range,
            **kwargs,
        )
    elif task == "MoveToCube-v0":
        return gym.make(
            task,
            box_half_size=env_args.cube_minsize,
            cube_rgba=env_args.cube_rgba[:3], 
            spawn_range=env_args.spawn_range,
            target_x=env_args.target_x,
            target_y=env_args.target_y,
            target_z=env_args.target_z,
            point_goal=env_args.point_goal,
            **kwargs,
        )
    elif task == "TurnFaucet-v0":
        return gym.make(
            task,
            model_ids="5007",
            **kwargs,
        )
    elif task in ["PickSingleYCB-v0", "CustomLiftYCB-v0"]:
        return gym.make(
            task,
            model_ids=env_args.object_name,
            **kwargs,
        )
        
    else:
        return gym.make(
            task,
            **kwargs,
        )
    