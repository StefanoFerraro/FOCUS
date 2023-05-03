from custom_maniskill_tasks.custom_lift import CustomLiftCubeEnv
from custom_maniskill_tasks.custom_stack import CustomStackCubeEnv
import gym

def make(task, env_args):
                
    kwargs = {
                "obs_mode": env_args.obs_mode,
                "reward_mode": env_args.reward_mode,
                "control_mode": env_args.control_mode,
                "camera_cfgs": {
                    "add_segmentation": True,
                    "height": env_args.size[0],
                    "width": env_args.size[1],
                    "texture_names": ("Color", "Position", "Segmentation"),}
            }
    
    if task == "CustomLiftCube-v0" or task == "CustomStackCube-v0":
        return gym.make(
            task,
            box_half_size=env_args.cube_minsize,
            cube_rgba=env_args.cube_rgba[:3], 
            spawn_range=env_args.spawn_range,
            **kwargs,
        )
    else:
        return gym.make(
            task,
            **kwargs,
        )
    