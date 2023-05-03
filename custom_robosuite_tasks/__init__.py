from custom_robosuite_tasks.custom_lift import CustomLift
from custom_robosuite_tasks.custom_stack import CustomStack
from robosuite.environments.base import REGISTERED_ENVS
import robosuite as suite


def make(task, env_args):

    kwargs = {
        "robots": "Panda",  # use Sawyer robot
        "has_renderer": False,  # on-screen renderer
        "has_offscreen_renderer": True,  # off-screen rendering needed for image obs
        "use_object_obs": True,  # provide object observations to agent
        "use_camera_obs": True,  # provide image observations to agent
        "camera_names": env_args.camera,
        "camera_depths": True,
        "camera_segmentations": env_args.segmentation_level,
        "camera_heights": env_args.size[0],  # image height
        "camera_widths": env_args.size[1],  # image width
        "horizon": env_args.horizon,  # each episode terminates after 200 steps
        "reward_shaping": True,
    }

    if task == "CustomLift" or task == "CustomStack":
        return CustomLift(
            **kwargs,
            cube_rgba=env_args.cube_rgba,
            cube_minsize=env_args.cube_minsize,
            spawn_range=env_args.spawn_range,
        )
    elif task in REGISTERED_ENVS:
        return suite.make(task, *kwargs)
    else:
        raise f"{task} not found"
