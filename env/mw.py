from .utils import *
from env.base_env import BaseEnv
import pickle
import gym
import pyquaternion as pq
import robosuite.utils.transform_utils as T
import mujoco
import metaworld
from copy import deepcopy
import custom_metaworld_tasks

class Metaworld(BaseEnv):
    def __init__(
        self,
        env_config,
        task="Lift",
        objs=["cube"],
        seed=None,
        action_repeat=1,
    ):
        super().__init__(env_config, task, objs, seed, action_repeat)
            
        # Construct the benchmark, sampling tasks
        if task == "robobin":
            self._env = custom_metaworld_tasks.RoboBinEnv(2)
        else:
            self.ml1 = metaworld.ML1(f"{task}-v2", seed=seed)
            env_cls = self.ml1.train_classes[f"{task}-v2"]
            self._env = env_cls()
        
        self._env._freeze_rand_vec = True
        self._env._last_rand_vec = [0, 0.9, 0]
        self._tasks = self.ml1.test_tasks
        self._env.camera_name = self.camera
        self._env.render_mode = "rgb_array"
        if task == "reach":
            with open(f"../../../mw_tasks/reach_harder/{seed}.pickle", "rb") as handle:
                self._tasks = pickle.load(handle)

        self.world_to_camera = self.get_camera_transform_matrix()
        self.camera_to_world = np.linalg.inv(self.world_to_camera)

        self.objects_pixels = [0] * len(objs)
        self.last_estimated_obj_pos = [[0, 0, 0]] * len(
            self.segmentation_instances
        )  # initialize to zero

        self.true_obj_pos, self.true_obj_ori = self.get_objects_pose()
        self.joints_names = [
            "l_close",
            "r_close",
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]

        self._duration = 0
        self.is_last = False
        self.main_object_id = [mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, part) for part in self.segmentation_instances[0]]

    def action_spec(
        self,
    ):
        return specs.BoundedArray(
            name="action",
            shape=self._env.action_space.shape,
            dtype=np.dtype("float32"),
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
        )

    @property
    def obs_space(self):
        spaces = self.common_obs_space
        spaces.update(
            {
                "proprio": gym.spaces.Box(
                    -5,
                    5,
                    [18],  # qpos + qvel * (7 joints + 2 fingers)
                    dtype=np.float32,
                ),
            }
        )
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        action.dtype = np.dtype("float32")
        return {"action": action}

    def generate_segmentation(self, rgb, include_background=True):
        channels = (
            len(self.segmentation_instances) + 1
            if include_background
            else len(self.segmentation_instances)
        )

        seg, _, _ = self.segmenter.generate(rgb, self.is_first)
        seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)

        seg_map = np.zeros(
            (channels, seg.shape[0], seg.shape[1]),
            dtype=np.uint8,
        )

        for i, instance in enumerate(self.segmentation_instances):
            seg_map[i][seg == i + 1] = 1

        if include_background:
            background_mask = np.all(seg_map == 0, axis=0)
            seg_map[-1][background_mask] = 1  # last layer is background layer

        return seg_map

    def get_proprio_data(self):
        proprio = []
        for jnt in self.joints_names:
            proprio.append(self._env.data.joint(jnt).qpos[0])
            proprio.append(self._env.data.joint(jnt).qvel[0])
        return proprio

    def get_objects_pose(self):
        obj_pos = {}
        obj_ori = {}
        for obj in self.segmentation_instances:
            obj_pos[obj[0]] = []
            obj_ori[obj[0]] = []
            for part in obj:
                obj_pos[obj[0]].append(self._env.data.body(part).xpos.copy())
                obj_ori[obj[0]].append(self._env.data.body(part).xquat.copy())

        return obj_pos, obj_ori

    def compute_displacements(self, true_objs_pos, true_objs_ori):
        true_pos_displacement = 0
        true_ori_displacement = 0
        true_vertical_displacement = 0

        for obj in self.segmentation_instances:
            for i in range(len(obj)):
                true_pos_displacement += np.sqrt(
                    np.sum(((true_objs_pos[obj[0]][i] - self.true_obj_pos[obj[0]][i]) ** 2))
                )

                true_ori_displacement += pq.Quaternion.absolute_distance(
                    pq.Quaternion(self.true_obj_ori[obj[0]][i]),
                    pq.Quaternion(true_objs_ori[obj[0]][i]),
                )
                true_vertical_displacement += abs(
                    true_objs_pos[obj[0]][i][2] - self.true_obj_pos[obj[0]][i][2]
                )

        return (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        )

    def get_camera_transform_matrix(self):
        # Cam Instrinsics
        fovy = self._env.model.camera(self.camera).fovy
        f = 0.5 * self.size[1] / np.tan(fovy * np.pi / 360)
        K = np.array([[f.item(), 0, self.size[0] / 2], [0, f.item(), self.size[1] / 2], [0, 0, 1]])

        # Cam Extrinsics
        camera_pos = self._env.model.camera(self.camera).pos
        camera_rot = self._env.model.camera(self.camera).mat0.reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        R = R @ camera_axis_correction

        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        return K_exp @ T.pose_inv(R)

    def step(self, action):
        self._duration += 1
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, _, _, info = self._env.step(action)
            if self.task == "hammer": # success metric worngly defined in native env, this is a workaround 
                info["success"] = True if (info["success"] and rew > 5.0) else False
                
            success += float(info["success"])
            reward += float(rew) if self.reward_shaping else float(info["success"])

        rgb = self._env.mujoco_renderer.render(
            render_mode="rgb_array", camera_name=self.camera
        )[::-1].copy()
        depth = self._env.mujoco_renderer.render(
            render_mode="depth_array", camera_name=self.camera
        )[::-1].copy()

        proprio = self.get_proprio_data()
        seg = self.generate_segmentation(rgb)

        # TODO check if state need to be modified

        contact = any([deepcopy(self.touching_object(part_id)) for part_id in self.main_object_id])
        new_true_obj_pos, new_true_obj_ori = self.get_objects_pose()
        (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        ) = self.compute_displacements(new_true_obj_pos, new_true_obj_ori)

        # objects_pos = self._env._get_pos_objects()
        self.true_obj_pos = new_true_obj_pos
        self.true_obj_ori = new_true_obj_ori

        objects_pos = self.pixel_to_world(seg, np.expand_dims(depth, axis=0))

        self.is_first = False

        if self._duration >= self.horizon:
            self.is_last = True

        obs = {
            "reward": reward,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "is_terminal": False,  # will be handled by per_episode function
            "rgb": cv2.resize(
                rgb, self.size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1),
            "depth": np.expand_dims(
                cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST), axis=0
            ),
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            # "state": state,
            "action": action,
            "success": bool(success),
            "in_areas": [False, False, False, False, False],
            "contact": contact,
            "pos_displacement": true_pos_displacement,
            "ang_displacement": true_ori_displacement,
            "vertical_displacement": true_vertical_displacement,
            "discount": 1,
        }
        return obs

    def reset(self):
        self._duration = 0
        self.is_last = False

        # Set task to ML1 choices
        task_id = np.random.randint(0, len(self._tasks))
        return self.reset_with_task_id(task_id)

    def reset_with_task_id(self, task_id):
        if self.camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.5, 0.3, 0.5]
        if self.camera == "robobin_custom":
            self._env.model.cam_pos[2][:] = [0.5, 0.3, 0.5]

        # Change table aspect (improve contract for object recognition)
        # self._env.model.material("table_wood").rgba = np.array([0.5, 0.5, 0.5, 1])
        # self._env.model.material("table_wood").reflectance = np.array([0.7])
        # self._env.model.material("table_wood").texid = np.array([-1])
        self.is_first = True

        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)

        _ = self._env.reset()[0]
        getattr(
            self, self.task.replace("-", "_") + "_init_pos"
        )()  # fix position of starting point of object (for segmentation purposes)

        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)

        rgb = self._env.mujoco_renderer.render(
            render_mode="rgb_array", camera_name=self.camera
        )[::-1].copy()
        depth = self._env.mujoco_renderer.render(
            render_mode="depth_array", camera_name=self.camera
        )[::-1].copy()

        proprio = self.get_proprio_data()

        seg = self.generate_segmentation(rgb)

        self.true_obj_pos, self.true_obj_ori = self.get_objects_pose()

        objects_pos = self.pixel_to_world(seg, np.expand_dims(depth, axis=0))

        obs = {
            "reward": 0.0,
            "is_first": self.is_first,
            "is_last": False,
            "is_terminal": False,
            "rgb": cv2.resize(
                rgb, self.size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1),
            "depth": np.expand_dims(
                cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST), axis=0
            ),
            "proprio": np.array(proprio).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            # "state": state,
            "action": np.zeros_like(self.act_space["action"].sample()),
            "success": False,
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": 0,
            "ang_displacement": 0,
            "vertical_displacement": 0,
            "discount": 1,
        }
        return obs

    def drawer_close_init_pos(self):
        self._env.obj_init_pos = np.array([0.082783, 0.9, 0])

        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = obj_init_pos
        # Set _target_pos to current drawer position (closed)
        self._target_pos = obj_init_pos + np.array([0.0, -0.16, 0.09])
        # Pull drawer out all the way and mark its starting position
        self._env._set_obj_xyz(-self._env.maxDist)
        self._env.obj_init_pos = self._env._get_pos_objects()
        mujoco.mj_forward(self._env.model, self._env.data)

    def drawer_open_init_pos(self):
        self._env.obj_init_pos = np.array([0.082783, 0.9, 0])
        # Set mujoco body to computed position
        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self._env.obj_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        self._env._target_pos = self._env.obj_init_pos + np.array(
            [0.0, -0.16 - self._env.maxDist, 0.09]
        )
        mujoco.mj_forward(self._env.model, self._env.data)

    def disassemble_init_pos(self):
        goal_pos = np.array([0.039741, 0.60054, 0.025005, 0.04004, 0.72229, 0.17005])
        self.obj_init_pos = goal_pos[:3]
        self._target_pos = goal_pos[-3:] + np.array([0.0, 0.0, 0.15])

        peg_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.08])

        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        ] = peg_pos
        self._env.model.site_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_SITE, "pegTop")
        ] = peg_top_pos

        mujoco.mj_forward(self._env.model, self._env.data)
        self._env._set_obj_xyz(self.obj_init_pos)

    def shelf_place_init_pos(self):
        goal_pos = np.array([0.046136, 0.58034, 0.019086, -0.078563, 0.84351, 0.29944])
        base_shelf_pos = goal_pos - np.array([0, 0, 0, 0, 0, 0.3])
        self._env.obj_init_pos = np.concatenate(
            (base_shelf_pos[:2], [self._env.obj_init_pos[-1]])
        )

        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "shelf")
        ] = base_shelf_pos[-3:]
        mujoco.mj_forward(self._env.model, self._env.data)
        self._target_pos = (
            self._env.model.site_pos[
                mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
            ]
            + self._env.model.body_pos[
                mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "shelf")
            ]
        )

        mujoco.mj_forward(self._env.model, self._env.data)
        self._env._set_obj_xyz(self._env.obj_init_pos)

    def handle_pull_init_pos(self):
        self._env.obj_init_pos = np.array([0.082783, 0.84197, 8.0383e-05])
        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self._env.obj_init_pos

        mujoco.mj_forward(self._env.model, self._env.data)
        self._env._set_obj_xyz(-0.1)
        self._target_pos = self._env._get_site_pos("goalPull")

    def door_open_init_pos(self):
        self._env.objHeight = self._env.data.geom("handle").xpos[2]

        self._env.obj_init_pos = np.array([0.091392, 0.89197, 0.15])
        self._env._target_pos = self._env.obj_init_pos + np.array([-0.3, -0.45, 0.0])

        self._env.model.body("door").pos = self._env.obj_init_pos
        self._env.model.site("goal").pos = self._env._target_pos

        mujoco.mj_forward(self._env.model, self._env.data)
        self._env._set_obj_xyz(0)
        self.maxPullDist = np.linalg.norm(
            self._env.data.geom("handle").xpos[:-1] - self._env._target_pos[:-1]
        )
        self._env.target_reward = 1000 * self._env.maxPullDist + 1000 * 2

    def door_close_init_pos(self):
        self._env.obj_init_pos = np.array([0.091392, 0.89197, 0.15])
        self._env._target_pos = self._env.obj_init_pos + np.array([0.2, -0.2, 0.0])

        self._env.model.body("door").pos = self._env.obj_init_pos
        self._env.model.site("goal").pos = self._env._target_pos
        
        mujoco.mj_forward(self._env.model, self._env.data)
        # keep the door open after resetting initial positions
        self._env._set_obj_xyz(-1.5708)
        
    def peg_insert_side_init_pos(self):
        pos_peg = np.array([0.14614, 0.66068, 0.02])
        pos_box = np.array([-0.33928, 0.53053, -0.00055824])
        self._env.obj_init_pos = pos_peg
        self._env.peg_head_pos_init = self._env._get_site_pos("pegHead")

        mujoco.mj_forward(self._env.model, self._env.data)
        self._env._set_obj_xyz(self._env.obj_init_pos)
        self._env.model.body_pos[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = pos_box
        self._env._target_pos = pos_box + np.array([0.03, 0.0, 0.13])

    def hammer_init_pos(self):
        self._env.hammer_init_pos = np.array([0.082783, 0.44197, 0])
        self._env.obj_init_pos = self._env.hammer_init_pos.copy()
        self._env._set_hammer_xyz(self._env.hammer_init_pos)

    def touching_object(self, object_geom_id):
        leftpad_geom_id = self._env.data.geom("leftpad_geom").id
        rightpad_geom_id = self._env.data.geom("rightpad_geom").id

        leftpad_object_contacts = [
            x
            for x in self._env.unwrapped.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self._env.unwrapped.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self._env.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self._env.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts
        )

        return 0 < leftpad_object_contact_force or 0 < rightpad_object_contact_force
