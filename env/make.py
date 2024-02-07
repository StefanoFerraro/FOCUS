from env.ms import PandaManiSkill
from env.rs import PandaRoboSuite
from env.mw import Metaworld
from env.dmc import DMCSuiteWrapper
from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ

import custom_dmc_tasks as cdmc
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels

from env.wrappers import *

env_classes = {"rs": PandaRoboSuite, "ms": PandaManiSkill, "mw": Metaworld}

def make(
    domain,
    task,
    action_repeat,
    seed,
    env_config=None,
):
    ''' General make function for environment initialization '''
    
    robot_envs = ["rs", "ms", "mw"]
    assert domain in robot_envs or domain == "dmc"
    
    objs = globals()[domain.upper() + "_TASKS_OBJ"][task]
    
    if domain in robot_envs:
        make_fn = _make
    elif domain == "dmc":
        make_fn = _make_dmc
    else:
        raise NotImplementedError
        
    env = make_fn(
        domain,
        task,
        objs,
        action_repeat,
        seed,
        env_config,
    )
    
    return env

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

def _make_dmc(
    domain,
    task,
    action_repeat,
    seed,
    env_config,
):
    visualize_reward = False
    subdomain, task = task.split("_", 1)
    horizon_steps = env_config.horizon
    img_size = env_config.renderer.size[0]
    
    # if task is not in the suite, use custom tasks
    if (subdomain, task) in suite.ALL_TASKS:
        env = suite.load(
            subdomain,
            task,
            task_kwargs=dict(random=seed, time_limit = horizon_steps * 0.02), # 0.02 is the timestep length
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    else:
        env = cdmc.make(
            subdomain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    # zoom in camera for quadruped
    camera_id = dict(quadruped=2).get(subdomain, 0)
    render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_kwargs)
    env._size = (img_size, img_size)
    env._camera = camera_id
    
    target_name = globals()["DMC_TASKS_OBJ"][task]
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return DMCSuiteWrapper(env, task, env_config, target_name, seed)

def _make_jaco(
    domain,
    task,
    frame_stack,
    action_repeat,
    seed,
    img_size,
):
    env = cdmc.make_jaco(task, "pixels", seed, img_size)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env

