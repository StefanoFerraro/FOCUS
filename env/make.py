from env.ms import PandaManiSkill
from env.rs import PandaRoboSuite
from env.mw import Metaworld
from env.dmc import DMCSuite
from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ

from env.wrappers import *
import env.custom_dmc_tasks as cdmc

env_classes = {"rs": PandaRoboSuite, "ms": PandaManiSkill, "mw": Metaworld, "dmc": DMCSuite}

def make(
    domain,
    task,
    action_repeat,
    seed,
    env_config=None,
):
    ''' General make function for environment initialization '''
    
    robot_envs = ["rs", "ms", "mw", "dmc"]
    assert domain in robot_envs 
    
    objs = globals()[domain.upper() + "_TASKS_OBJ"][task]
    
    if domain in robot_envs:
        make_fn = _make
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

