import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose

from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv


@register_env("CustomLiftYCB-v0", max_episode_steps=200, override=True)
class CustomLiftYCBEnv(PickSingleYCBEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_height = 0.2

    def evaluate(self, **kwargs):
        obj_to_goal_dist = self.goal_height - self.obj_pose.p[2]
        is_obj_placed = self.goal_height < self.obj_pose.p[2]
        is_robot_static = self.check_robot_static()
        return dict(
            obj_to_goal_pos=obj_to_goal_dist,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )

    def compute_dense_reward(self, info, **kwargs):

        # Sep. 14, 2022:
        # We changed the original complex reward to simple reward,
        # since the original reward can be unfriendly for RL,
        # even though MPC can solve many objects through the original reward.

        reward = 0.0

        if info["success"]:
            reward = 10.0
        else:
            obj_pose = self.obj_pose

            # reaching reward
            tcp_wrt_obj_pose = obj_pose.inv() * self.tcp.pose
            tcp_to_obj_dist = np.linalg.norm(tcp_wrt_obj_pose.p)
            reaching_reward = 1 - np.tanh(
                3.0
                * np.maximum(
                    tcp_to_obj_dist - np.linalg.norm(self.model_bbox_size), 0.0
                )
            )
            reward = reward + reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
            reward += 3.0 if is_grasped else 0.0

            # reaching-goal reward
            if is_grasped:
                obj_to_goal_dist = self.goal_height - obj_pose.p[2]
                reaching_goal_reward = 3 * (
                    1 - np.tanh(3.0 * obj_to_goal_dist)
                )
                reward += reaching_goal_reward

        return reward
