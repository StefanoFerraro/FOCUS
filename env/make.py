from env.ms import PandaManiSkill
from env.rs import PandaRoboSuite
from env.mw import Metaworld
from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ

import custom_dmc_tasks as cdmc
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels

from env.wrappers import *


env_classes = {"rs": PandaRoboSuite, "ms": PandaManiSkill, "mw": Metaworld}


def _make(
    env_type,
    task,
    objs,
    action_repeat,
    seed,
    env_config,
):
    env_class = env_classes[env_type]

    return env_class(env_config, task, objs, seed, action_repeat)


def _make_jaco(
    obs_type,
    domain,
    task,
    frame_stack,
    action_repeat,
    seed,
    img_size,
):
    env = cdmc.make_jaco(task, obs_type, seed, img_size)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(
    obs_type,
    domain,
    task,
    frame_stack,
    action_repeat,
    seed,
    img_size,
):
    visualize_reward = False
    domain, task = task.split("_")
    
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    else:
        env = cdmc.make(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == "pixels":
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(
    domain,
    task,
    obs_type,
    frame_stack,
    action_repeat,
    seed,
    env_config=None,
):
    assert obs_type in ["states", "pixels"]
    # domain = dict(cup="ball_in_cup", point="point_mass").get(domain, domain)

    obj_envs = ["rs", "ms", "mw"]

    if domain in obj_envs:
        make_fn = _make
        objs = globals()[domain.upper() + "_TASKS_OBJ"][task]
        env_type = domain[:2]

        env = make_fn(
            env_type,
            task,
            objs,
            action_repeat,
            seed,
            env_config,
        )
        return env

    else:
        make_fn = _make_jaco if domain == "jaco" else _make_dmc
    
    env = make_fn(
        obs_type,
        domain,
        task,
        frame_stack,
        action_repeat,
        seed,
        64,
    )

    if obs_type == "pixels":
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)

    return DreamerObsWrapper(env)
