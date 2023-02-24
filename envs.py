from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os
import cv2

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import custom_dmc_tasks as cdmc
import gym
import pickle

import robosuite as suite
from robosuite.wrappers import Wrapper
from gym.core import Env
from gym import spaces


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if "front_close" in wrapped_obs_spec:
            spec = wrapped_obs_spec["front_close"]
            # drop batch dim
            self._obs_spec["pixels"] = specs.BoundedArray(
                shape=spec.shape[1:],
                dtype=spec.dtype,
                minimum=spec.minimum,
                maximum=spec.maximum,
                name="pixels",
            )
            wrapped_obs_spec.pop("front_close")

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter(
                (
                    np.int(np.prod(spec.shape))
                    for spec in wrapped_obs_spec.values()
                ),
                np.int32,
            )
        )

        self._obs_spec["observations"] = specs.Array(
            shape=(dim,), dtype=np.float32, name="observations"
        )

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if "front_close" in time_step.observation:
            pixels = time_step.observation["front_close"]
            time_step.observation.pop("front_close")
            pixels = np.squeeze(pixels)
            obs["pixels"] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs["observations"] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()["observations"]
        self._obs_spec = specs.Array(
            wrapped_obs_spec.shape, dtype, "observation"
        )

    def _transform_observation(self, time_step):
        obs = time_step.observation["observations"].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DreamerObsWrapper:
    def __init__(self, env):
        self._env = env
        self._ignored_keys = []

    @property
    def obs_space(self):
        spaces = {
            "observation": self._env.observation_spec(),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(
            (spec.minimum) * spec.shape[0],
            (spec.maximum) * spec.shape[0],
            shape=spec.shape,
            dtype=np.float32,
        )
        return {"action": action}

    def step(self, action):
        # assert np.isfinite(action['action']).all(), action['action']
        time_step = self._env.step(action)
        assert time_step.discount in (0, 1)
        obs = {
            "reward": time_step.reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "observation": time_step.observation,
            "action": action,
            "discount": time_step.discount,
        }
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": time_step.observation,
            "action": np.zeros_like(self.act_space["action"].sample()),
            "discount": time_step.discount,
        }
        return obs

    def __getattr__(self, name):
        if name == "obs_space":
            return self.obs_space
        if name == "act_space":
            return self.act_space
        return getattr(self._env, name)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        ts, obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            ts = dm_env.TimeStep(
                dm_env.StepType.LAST, ts.reward, ts.discount, ts.observation
            )
            obs["is_last"] = True
            self._step = None
        return ts, obs

    def reset(self):
        self._step = 0
        return self._env.reset()

    # def reset_with_task_id(self, task_id):
    #     self._step = 0
    #     return self._env.reset_with_task_id(task_id)


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (
            self._high - self._low
        ) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class PandaGymWrapper(Wrapper, Env):
    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join(
            [type(robot.robot_model).__name__ for robot in self.env.robots]
        )
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        # if keys is None:
        #     keys = []
        #     # Add object obs if requested
        #     if self.env.use_object_obs:
        #         keys += ["object-state"]
        #     # Add image obs if requested
        #     if self.env.use_camera_obs:
        #         keys += [
        #             f"{cam_name}_image" for cam_name in self.env.camera_names
        #         ]
        #     # Iterate over all robots to add to state
        #     for idx in range(len(self.env.robots)):
        #         keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()

        # filter state based on keys
        obs = {key: obs[key] for key in self.keys}

        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)


class PandaRoboSuite:
    def __init__(
        self, task="Stack", seed=None, action_repeat=1, size=(64, 64)
    ):
        os.environ["MUJOCO_GL"] = "egl"
        self._camera = "agentview"

        self._proprio_keys = ["robot0_proprio-state"]

        self._obs_keys = [
            self._camera + "_image",
            self._camera + "_depth",
        ]
        self.segmentation_instances = [
            "cube",
            "Panda0",
            "PandaGripper0",
        ]
        self._object_keys = ["object-state"]
        self._env = PandaGymWrapper(
            suite.make(
                task,
                robots="Panda",  # use Sawyer robot
                has_renderer=False,  # on-screen renderer
                has_offscreen_renderer=True,  # off-screen rendering needed for image obs
                use_object_obs=True,  # provide object observations to agent
                use_camera_obs=True,  # provide image observations to agent
                camera_names=self._camera,
                camera_depths=True,
                camera_segmentations="element",
                camera_heights=size[0],  # image height
                camera_widths=size[1],  # image width
                horizon=250,  # each episode terminates after 200 steps
                reward_shaping=True,
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
            ),
            keys=self._proprio_keys + self._obs_keys,
        )

        self._size = size
        self._action_repeat = action_repeat
        self._seed = seed
        self._task = task

    def segmentation_channel_split(self, seg):

        instances_to_ids = self._env.model.instances_to_ids
        seg_map = np.zeros(
            (len(self.segmentation_instances) + 1, seg.shape[1], seg.shape[2]),
            dtype=np.uint8,
        )

        for i, instance in enumerate(self.segmentation_instances):
            for id in instances_to_ids[instance]["geom"]:
                seg_map[i][seg[0] == id] = 1

        panda_idx = self.segmentation_instances.index("Panda0")
        non_panda_idx = [
            x
            for x in range(len(self.segmentation_instances))
            if x != panda_idx
        ]
        panda_median_layer = cv2.medianBlur(
            seg_map[panda_idx], 3
        )  # median filtering for panda segmentation

        # check that in panda layer there is no overlap with other encodings
        panda_mask = np.all(seg_map[non_panda_idx] == 0, axis=0)
        seg_map[panda_idx][panda_mask] = panda_median_layer[panda_mask]

        background_mask = np.all(seg_map == 0, axis=0)
        seg_map[-1][background_mask] = 1  # last layer is background layer

        return seg_map

    @staticmethod
    def image_segmentation(image, seg):
        """
        Segmentation of image based on segmentation mask, works both with rgb and depth images.
        seg: (dim1, dim2, channels)
        image: (dim1, dim2, 1/3)
        
        Return:
        seg_image: (dim1, dim2, 1/3, channels)
        """

        seg_chs, _, _ = seg.shape
        image_chs, _, _ = image.shape

        seg_mask = np.repeat(np.expand_dims(seg, axis=1), image_chs, 1)
        seg_image = np.zeros((seg_chs, *image.shape), dtype=image.dtype)

        for ch in range(seg_chs):
            seg_image[ch] = image * seg_mask[ch]

        return seg_image

    def rgb_spec(self,):
        v = self.obs_space["rgb"]
        return specs.BoundedArray(
            name="rgb",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def depth_spec(self,):
        v = self.obs_space["depth"]
        return specs.BoundedArray(
            name="depth",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def proprio_spec(self,):
        v = self.obs_space["proprio"]
        return specs.BoundedArray(
            name="proprio",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def segmentation_spec(self,):
        v = self.obs_space["segmentation"]
        return specs.BoundedArray(
            name="segmentation",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def seg_rgb_spec(self,):
        v = self.obs_space["seg_rgb"]
        return specs.BoundedArray(
            name="seg_rgb",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def seg_depth_spec(self,):
        v = self.obs_space["seg_depth"]
        return specs.BoundedArray(
            name="seg_depth",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def action_spec(self,):
        return specs.BoundedArray(
            name="action",
            shape=self._env.action_space.shape,
            dtype=self._env.action_space.dtype,
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
        )

    @property
    def obs_space(self):
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "depth": gym.spaces.Box(
                -np.inf, np.inf, (1,) + self._size, dtype=np.float32,
            ),
            "proprio": gym.spaces.Box(
                -np.inf,
                np.inf,
                self._env.modality_dims["robot0_proprio-state"],
                dtype=np.float32,
            ),
            "segmentation": gym.spaces.Box(
                0,
                1,
                (len(self.segmentation_instances) + 1,) + self._size,
                dtype=np.uint8,
            ),
            "seg_rgb": gym.spaces.Box(
                0,
                255,
                (len(self.segmentation_instances) + 1, 3,) + self._size,
                dtype=np.uint8,
            ),
            "seg_depth": gym.spaces.Box(
                -np.inf,
                np.inf,
                (len(self.segmentation_instances) + 1, 1,) + self._size,
                dtype=np.float32,
            ),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def _proprio_obs(self, state):
        proprio = []

        for key in self._proprio_keys:
            proprio += list(state[key])

        return proprio

    def _state_extraction(self, env_state):
        proprio = self._proprio_obs(env_state)

        rgb = env_state[self._camera + "_image"][::-1].transpose(2, 0, 1)
        depth = env_state[self._camera + "_depth"][::-1].transpose(2, 0, 1)

        seg = env_state[self._camera + "_segmentation_element"][
            ::-1
        ].transpose(2, 0, 1)

        state = {}
        for key in self._proprio_keys or self._obs_keys:
            state[key] = env_state[key]

        return proprio, rgb, depth, seg, state

    def step(self, action):
        # assert np.isfinite(action["action"]).all(), action["action"]
        # TODO: check state match with observation
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            env_state, rew, done, info = self._env.step(action)
            success += float(done)
            reward += float(rew)
        success = min(success, 1.0)
        assert success in [0.0, 1.0]

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)

        seg = self.segmentation_channel_split(seg)

        seg_rgb = self.image_segmentation(rgb, seg)
        seg_depth = self.image_segmentation(depth, seg)

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": done,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "segmentation": seg,
            "seg_rgb": seg_rgb,
            "seg_depth": seg_depth,
            "state": self._env._flatten_obs(state),
            "action": action,
            "success": success,
            "discount": 1,
        }

        return obs

    def reset(self):
        env_state = self._env.reset()

        proprio, rgb, depth, seg, state = self._state_extraction(env_state)
        seg = self.segmentation_channel_split(seg)

        seg_rgb = self.image_segmentation(rgb, seg)
        seg_depth = self.image_segmentation(depth, seg)

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "rgb": rgb,
            "depth": depth,
            "proprio": np.array(proprio).astype(np.float32),
            "segmentation": seg,
            "seg_rgb": seg_rgb,
            "seg_depth": seg_depth,
            "state": self._env._flatten_obs(state),
            "action": np.zeros_like(self.act_space["action"].sample()),
            "success": False,
            "discount": 1,
        }

        return obs

    # def reset_with_task_id(self, task_id):
    # if self._camera == "corner2":
    #     self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]

    # task = self._tasks[task_id]
    # self._env.set_task(task)

    # state = self._env.reset()
    # # This ensures the first observation is correct in the renderer
    # self._env.sim.render(
    #     *self._size, mode="offscreen", camera_name=self._camera
    # )
    # for site in self._env._target_site_config:
    #     self._env._set_pos_site(*site)
    # self._env.sim._render_context_offscreen._set_mujoco_buffers()

    # obs = {
    #     "reward": 0.0,
    #     "is_first": True,
    #     "is_last": False,
    #     "is_terminal": False,
    #     "observation": self._env.sim.render(
    #         *self._size, mode="offscreen", camera_name=self._camera
    #     )
    #     .transpose(2, 0, 1)
    #     .copy(),
    #     "state": state,
    #     "action": np.zeros_like(self.act_space["action"].sample()),
    #     "success": False,
    #     "discount": 1,
    # }
    # return (
    #     dm_env.TimeStep(
    #         step_type=dm_env.StepType.FIRST,
    #         reward=None,
    #         discount=None,
    #         observation=obs["observation"],
    #     ),
    #     obs,
    # )


def _make_panda(
    obs_type, domain, task, frame_stack, action_repeat, seed, img_size,
):
    env = PandaRoboSuite("Lift", seed, action_repeat, (img_size, img_size))
    return env


def _make_jaco(
    obs_type, domain, task, frame_stack, action_repeat, seed, img_size,
):
    env = cdmc.make_jaco(task, obs_type, seed, img_size,)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(
    obs_type, domain, task, frame_stack, action_repeat, seed, img_size,
):
    visualize_reward = False
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
        render_kwargs = dict(
            height=img_size, width=img_size, camera_id=camera_id
        )
        env = pixels.Wrapper(
            env, pixels_only=True, render_kwargs=render_kwargs
        )
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(
    name, obs_type, frame_stack, action_repeat, seed, img_size=84,
):
    assert obs_type in ["states", "pixels"]
    domain, task = name.split("_", 1)
    domain = dict(cup="ball_in_cup", point="point_mass").get(domain, domain)

    if domain == "panda":
        make_fn = _make_panda
        env = make_fn(
            obs_type, domain, task, frame_stack, action_repeat, seed, img_size,
        )
        return env
    else:
        make_fn = _make_jaco if domain == "jaco" else _make_dmc
    env = make_fn(
        obs_type, domain, task, frame_stack, action_repeat, seed, img_size,
    )

    if obs_type == "pixels":
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)

    return DreamerObsWrapper(env)
