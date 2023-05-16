from typing import List, Tuple
import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose

from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.stack_cube import (
    StackCubeEnv,
    UniformSampler,
)


@register_env("CustomStackCube-v0", max_episode_steps=200, override=True)
class CustomStackCubeEnv(StackCubeEnv):
    def __init__(
        self,
        *args,
        obj_init_rot_z=True,
        box_half_size=[0.02, 0.02, 0.02],
        cube_rgba=(1, 0, 0),
        spawn_range=[-0.1, 0.1],
        **kwargs,
    ):

        self.obj_init_rot_z = obj_init_rot_z
        self.box_half_size = np.array(box_half_size, np.float32)
        self.cube_rgba = cube_rgba
        self.spawn_range = spawn_range

        super().__init__(*args, **kwargs)

        self.box_half_size = np.array(
            box_half_size, np.float32
        )  # reset for avoiding confilict with parent init

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.cubeA = self._build_cube(
            self.box_half_size, color=self.cube_rgba, name="cubeA"
        )
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(*self.spawn_range, [2])

        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.box_half_size[:2]) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.box_half_size[2]
        cubeA_pose = Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)
