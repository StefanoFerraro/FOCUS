import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose

from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import LiftCubeEnv


@register_env("LiftCube-v0", max_episode_steps=200, override=True)
class CustomLiftCubeEnv(LiftCubeEnv):
    def __init__(
        self,
        *args,
        obj_init_rot_z=True,
        box_half_size=[0.02, 0.02, 0.02],
        cube_rgba=(1, 0, 0),
        spawn_range=[-0.1, 0.1],
        **kwargs
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
        self.obj = self._build_cube(self.box_half_size, color=self.cube_rgba)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        x = self._episode_rng.uniform(*self.spawn_range, [1])
        y = self._episode_rng.uniform(-0.025, 0.025, [1])
        xyz = np.hstack([x, y, self.box_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))
