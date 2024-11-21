from env.custom_robosuite_tasks.custom_lift import CustomLift
from env.custom_robosuite_tasks.custom_stack import CustomStack
from env.custom_robosuite_tasks.move_to import MoveTo
from robosuite.environments.base import REGISTERED_ENVS

import robosuite as suite
from robosuite.controllers import load_controller_config


def make(task, env_args):

    controller_config = load_controller_config(
        default_controller=env_args.controller
    )
    if env_args.controller == "OSC_POSE":
        controller_config["orientation_limits"] = [0, 0]

    kwargs = {
        "robots": "Panda",  # use Sawyer robot
        "controller_configs": controller_config,
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
        "reward_shaping": env_args.reward_shaping,
    }

    if task == "Lift":
        return CustomLift(
            **kwargs,
            cube_rgba=env_args.cube_rgba,
            cube_minsize=env_args.cube_minsize,
            spawn_range=env_args.spawn_range,
        )
    elif task == "Stack":
        return CustomStack(
            **kwargs,
            cube_rgba=env_args.cube_rgba,
            cube_minsize=env_args.cube_minsize,
            spawn_range=env_args.spawn_range,
            random_placement=env_args.random_placement,
        )
    elif task == "MoveTo":
        return MoveTo(
            **kwargs,
            cube_rgba=env_args.cube_rgba,
            cube_minsize=env_args.cube_minsize,
            spawn_range=env_args.spawn_range,
            target_x=env_args.target_x,
            target_y=env_args.target_y,
            target_z=env_args.target_z,
            point_goal=env_args.point_goal,
        )
    elif task in REGISTERED_ENVS:
        return suite.make(task, **kwargs)
    else:
        raise f"{task} not found"
