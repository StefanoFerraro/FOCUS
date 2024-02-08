import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose

from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import LiftCubeEnv


@register_env("MoveToCube-v0", max_episode_steps=200, override=True)
class MoveToEnv(LiftCubeEnv):
    def __init__(
        self,
        *args,
        obj_init_rot_z=True,
        box_half_size=[0.02, 0.02, 0.02],
        cube_rgba=(1, 0, 0),
        spawn_range=[-0.1, 0.1],
        target_x=None,
        target_y=None,
        target_z=None,
        point_goal=False,
        **kwargs,
    ):
        self.obj_init_rot_z = obj_init_rot_z
        self.box_half_size = np.array(box_half_size, np.float32)
        self.cube_rgba = cube_rgba
        self.spawn_range = spawn_range

        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.point_goal = point_goal

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

    def _initialize_task(
        self,
        max_trials=100,
        verbose=False,
    ):

        if self.target_x != "None" and self.target_y != "None" and self.target_z != "None":
            self.goal_pos = np.hstack([self.target_x, self.target_y, self.target_z])
        else:
            self.goal_pos = [0,0,0]
            
        self.goal_site.set_pose(Pose(self.goal_pos))

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    @staticmethod
    def is_in_area(val, target):
        return abs(val) > abs(target) and np.sign(val) == np.sign(target)

    def check_obj_in_area(self):
        
        x_in_goal = True
        y_in_goal = True
        z_in_goal = True
        
        if self.target_x != "None":
            x_in_goal = self.is_in_area(self.obj.pose.p[0], self.target_x)
        if self.target_y != "None":
            y_in_goal = self.is_in_area(self.obj.pose.p[1], self.target_y)
        if self.target_z != "None":
            z_in_goal = self.is_in_area(self.obj.pose.p[2], self.target_z)
            
        return x_in_goal and y_in_goal and z_in_goal

    def evaluate(self, **kwargs):
        if self.point_goal:
            is_obj_placed = self.check_obj_placed()
            # is_robot_static = self.check_robot_static()
        else:
            is_obj_placed = self.check_obj_in_area()
            #is_robot_static = True # not relevant for this task
            
        return dict(
            is_obj_placed=is_obj_placed,
            # is_robot_static=is_robot_static,
            success=is_obj_placed #and is_robot_static,
        )
        