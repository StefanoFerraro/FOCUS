from .utils import *
from env.base_envs import ObjectsEnv
import pickle
import gym
import pyquaternion as pq
import robosuite.utils.transform_utils as T
import mujoco
import metaworld
from copy import deepcopy
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from metaworld.envs.mujoco.utils import reward_utils

limits_exploration_area = {"bin-picking": [[-0.2, -0.1, 0], [0.2, 0.3, 0.15]],
                           "shelf-place": [[-0.35, -0.1, 0], [0.3, 0.4, 0.3]]}

class Metaworld(ObjectsEnv):
    def __init__(
        self,
        env_config,
        task="Lift",
        objs=["cube"],
        seed=None,
        action_repeat=1,
    ):
        super().__init__(env_config, task, objs, seed, action_repeat)
        
        if self.task in limits_exploration_area.keys(): self.limits_exploration_area = env_config.limits_exploration_area = limits_exploration_area[self.task] 
        else: self.limits_exploration_area = env_config.limits_exploration_area = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        self._make()

    def _make(self):
        self.ml1 = metaworld.ML1(f"{self.task}-v2", seed=self.seed)
        env_cls = self.ml1.train_classes[f"{self.task}-v2"]
        self._env = env_cls()
        
        self._env._freeze_rand_vec = True
        self._env._last_rand_vec = [0, 0.9, 0]
        self._tasks = self.ml1.test_tasks
        self._env.camera_name = self.camera
        self._env.render_mode = "rgb_array"
        if self.task == "reach":
            with open(f"../../../mw_tasks/reach_harder/{self.seed}.pickle", "rb") as handle:
                self._tasks = pickle.load(handle)

        self.world_to_camera = self.get_camera_transform_matrix()
        self.camera_to_world = np.linalg.inv(self.world_to_camera)

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
        
        self.seg_render = OffScreenViewer(self._env.model, self._env.data)
        self.camera_id = self._env.model.camera(self.camera).id
        self.true_obj_pos, self.true_obj_ori = self.get_objects_pose()
        
        # if self.task == "bin-picking":
            # self._env.set_xyz_action = self.custom_set_xyz_action # swap the set_xyz_action function to allow for custom action space
            # self._env._reset_hand = self.custom_reset_hand

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
                "target": gym.spaces.Box(
                    -1,
                    1,
                    (3,), # 3 states for the target
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

    def generate_segmentation(self, rgb):
        if self.gt_segmentation:
            seg = cv2.resize(self.seg_render.render(render_mode="rgb_array", camera_id=self.camera_id, segmentation=True), self.size, interpolation=cv2.INTER_NEAREST)[::-1]
            seg = seg[:,:,1]
        else:  
            if self.task=="bin-picking":
                im = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  
            else:
                im = rgb
            seg = self.segmenter.generate(im, self.is_first)
            seg = cv2.resize(seg, self.size, interpolation=cv2.INTER_NEAREST)
        return seg

    def segmentation_channel_split(self, seg, include_background=False):
        seg_map = np.zeros((self.seg_channels, seg.shape[0], seg.shape[1]), dtype=np.uint8)

        if self.gt_segmentation:           
            for i, _ in enumerate(self.segmentation_instances):
                obj_id = self._env.model.body(self.segmentation_instances[i][0]).geomadr
                seg_map[i][seg==obj_id] = 1
        else:
            for i, _ in enumerate(self.segmentation_instances):
                seg_map[i][seg == i + 1] = 1
    
        seg_map = self.seg_background(seg_map, include_background)

        return seg_map
    
    def _state_generation(self):
        rgb = self._env.mujoco_renderer.render(
            render_mode="rgb_array", camera_name=self.camera
        )[::-1].copy()
        depth = self._env.mujoco_renderer.render(
            render_mode="depth_array", camera_name=self.camera
        )[::-1].copy()

        proprio = self.get_proprio_data()
        seg = self.generate_segmentation(rgb)
        return proprio, rgb, depth, seg
    
    def get_proprio_data(self):
        proprio = []
        for jnt in self.joints_names:
            proprio.append(self._env.data.joint(jnt).qpos[0])
            proprio.append(self._env.data.joint(jnt).qvel[0])
        return proprio

    def get_objects_pose(self):
        obj_pos = {}
        obj_ori = {}
        
        # implementation for object with multiple parts
        # for obj in self.segmentation_instances:
        #     obj_pos[obj[0]] = []
        #     obj_ori[obj[0]] = []
        #     for part in obj:
        #         obj_pos[obj[0]].append(self._env.data.body(part).xpos.copy())
        #         obj_ori[obj[0]].append(self._env.data.body(part).xquat.copy())        
        # obj_pos[obj[0]] = self._env._get_pos_objects().copy() - self.object_start_pos #remove fixed offset 
        # obj_ori[obj[0]] = self._env._get_quat_objects().copy()  
        
        obj_pos[self.target_obj[0]] = self._env.data.body(self.segmentation_instances[0][0]).xpos.copy() - self.object_start_pos
        obj_ori[self.target_obj[0]] = self._env.data.body(self.segmentation_instances[0][0]).xquat.copy()    

        return obj_pos, obj_ori

    def compute_displacements(self, true_objs_pos, true_objs_ori):
        true_pos_displacement = 0
        true_ori_displacement = 0
        true_vertical_displacement = 0

        for obj in self.segmentation_instances:
            true_pos_displacement += np.sqrt(
                np.sum(((true_objs_pos[obj[0]] - self.true_obj_pos[obj[0]]) ** 2))
            )

            true_ori_displacement += pq.Quaternion.absolute_distance(
                pq.Quaternion(self.true_obj_ori[obj[0]]),
                pq.Quaternion(true_objs_ori[obj[0]]),
            )
            true_vertical_displacement += abs(
                true_objs_pos[obj[0]][2] - self.true_obj_pos[obj[0]][2]
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

    def compute_distance_to_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self._env.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._env._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = np.linalg.norm(self._env.obj_init_pos - target)

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        object_grasped = self._env._gripper_caging_reward(
            action=action,
            obj_pos=obj,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=False,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if (
            0.0 < obj[2] < 0.24
            and (target[0] - 0.15 < obj[0] < target[0] + 0.15)
            and ((target[1] - 3 * _TARGET_RADIUS) < obj[1] < target[1])
        ):
            z_scaling = (0.24 - obj[2]) / 0.24
            y_scaling = (obj[1] - (target[1] - 3 * _TARGET_RADIUS)) / (
                3 * _TARGET_RADIUS
            )
            bound_loss = reward_utils.hamacher_product(y_scaling, z_scaling)
            in_place = np.clip(in_place - bound_loss, 0.0, 1.0)

        if (
            (0.0 < obj[2] < 0.24)
            and (target[0] - 0.15 < obj[0] < target[0] + 0.15)
            and (obj[1] > target[1])
        ):
            in_place = 0.0

        if (
            tcp_to_obj < 0.025
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > self._env.obj_init_pos[2])
        ):
            reward += 1.0 + 5.0 * in_place

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.0
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]

    def custom_set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self._env.action_scale
        new_mocap_pos = self._env.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self._env.mocap_low,
            self._env.mocap_high,
        )
        self._env.data.mocap_pos = new_mocap_pos        
        self._env.data.mocap_quat = (pq.Quaternion([1, 0, 1, 0]) * pq.Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)).elements
        
    def custom_reset_hand(self, steps=50):
        mocap_id = self._env.model.body_mocapid[self._env.data.body("mocap").id]
        for _ in range(steps):
            self._env.data.mocap_pos[mocap_id][:] = self._env.hand_init_pos
            self._env.data.mocap_quat[mocap_id][:] = (pq.Quaternion([1, 0, 1, 0]) * pq.Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)).elements
            self._env.do_simulation([-1, 1], self._env.frame_skip)
        self._env.init_tcp = self._env.tcp_center
        self._env.init_tcp = self._env.tcp_center
        
    def step(self, action):
        self._duration += 1
        reward = 0.0
        success = 0.0
        for _ in range(self.action_repeat):
            state, rew, _, _, info = self._env.step(action)
            if self.task == "hammer": # success metric worngly defined in native env, this is a workaround 
                info["success"] = True if (info["success"] and rew > 5.0) else False
            
            if self.task == "bin-picking": # in case of reward as distance to the chosen target
                (
                rew,
                tcp_to_obj,
                tcp_open,
                obj_to_target,
                grasp_reward,
                in_place,
                )  = self.compute_distance_to_reward(action, state)
                info["success"] = float(obj_to_target <= 0.07)
                
            success += float(info["success"])
            reward += float(rew) if self.reward_shaping else float(info["success"])

        proprio, rgb, depth, seg = self._state_generation()
        
        seg = self.segmentation_channel_split(seg, self.include_background)

        # contact = any([deepcopy(self.touching_object(part_id)) for part_id in self.main_object_id])
        new_true_obj_pos, new_true_obj_ori = self.get_objects_pose()
        (
            true_pos_displacement,
            true_ori_displacement,
            true_vertical_displacement,
        ) = self.compute_displacements(new_true_obj_pos, new_true_obj_ori)

        # objects_pos = self._env._get_pos_objects()
        self.true_obj_pos = new_true_obj_pos
        self.true_obj_ori = new_true_obj_ori

        # objects_pos = self.pixel_to_world(seg, np.expand_dims(depth, axis=0))
        objects_pos = [self.true_obj_pos[self.target_obj[0]]]
        
        object_to_target = objects_pos[0] - (self._env._target_pos)
        if self.dist_as_rw:
            reward = - np.linalg.norm(object_to_target)

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
            "target": np.array(object_to_target).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
            "action": action,
            "success": bool(success),
            "in_areas": [False, False, False, False, False],
            "contact": False,
            "pos_displacement": true_pos_displacement,
            "ang_displacement": true_pos_displacement,
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
        if self.task == "bin-picking": # get the camera closer
            self._env.model.cam_pos[self.camera_id][:] = [0, 1.1, .5]
            self._env.model.cam_quat[self.camera_id][:] = (pq.Quaternion([-1, 0, 0, 0]) * pq.Quaternion(axis=[1.0, 0.0, 0.0], degrees=-45)).elements

        self.is_first = True

        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)

        reward = 0.0
        _ = self._env.reset()[0]
        if self.task != "bin-picking":
            self._env.model.site("goal").rgba[-1] = 0 # hide the goal in the scene
        
        getattr(
            self, self.task.replace("-", "_") + "_init_pos"
        )()  # fix position of starting point of object (for segmentation purposes)

        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)

        proprio, rgb, depth, seg = self._state_generation()

        seg = self.segmentation_channel_split(seg, self.include_background)

        self.true_obj_pos, self.true_obj_ori = self.get_objects_pose()

        # objects_pos = self.pixel_to_world(seg, np.expand_dims(depth, axis=0))
        objects_pos = [self.true_obj_pos[self.target_obj[0]]]

        object_to_target = objects_pos[0] - (self._env._target_pos)
        if self.dist_as_rw:
            reward = - np.linalg.norm(object_to_target)
        
        obs = {
            "reward": reward,
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
            "target": np.array(object_to_target).astype(np.float32),
            "objects_pos": np.array(objects_pos).astype(np.float32),
            "segmentation": seg,
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
        ] = self._env.obj_init_pos
        # Set _target_pos to current drawer position (closed)
        self._target_pos = self._env.obj_init_pos + np.array([0.0, -0.16, 0.09])
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
        random_init = np.random.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0.025])
        self._env._set_obj_xyz(self._env.obj_init_pos + random_init)

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

    def bin_picking_init_pos(self):
        # self._env.model.material('obj_green').rgba = [1,1,0,1] 
        mujoco.mj_forward(self._env.model, self._env.data)
        self._env.obj_init_pos = np.array([  -0.125,     0.7,    0.03])
        random_init = np.random.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0.01])
        self._env._set_obj_xyz(self._env.obj_init_pos + random_init)

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

    #### ADDED FUNCTIONS ####
    
    def _target_hide(self):
        if self.task != "bin-picking":
            self._env.model.site("goal").rgba[-1] = 0 # hide the goal in the scene
        
    def _target_show(self):
        if self.task != "bin-picking":
            self._env.model.site("goal").rgba[-1] = 0.5 # hide the goal in the scene
                
    def set_target(self, target_pos): 
        # set target in env (visualization purposes)
        self._env._target_pos = target_pos + np.array(self.object_start_pos)   
        if self.task != "bin-picking":
            self._env.model.site("goal").pos = target_pos + np.array(self.object_start_pos)   
    
    def get_rgb_with_target(self, target=None):
        # in case of dmc manipulator environment, the target position needs to update at every step, given the internal machanics
        self._target_show()
        target_rgb = self.render()
        self._target_hide()
        return target_rgb
    
    def get_goals(self):
        return self.set_goals_for_task()
  
    def set_goals_for_task(self):
        if self.task in ["bin-picking"]:
            back_left = [[-0.2, -0.05, 0.03], [-0.2, -0.05, 0.03]]
            back_center = [[0, -0.05, 0.03], [0, -0.05, 0.03]]
            back_right = [[0.2, -0.05, 0.03], [0.2, -0.05, 0.03]]
            in_between_boxes = [[0, 0.2, 0.1], [0, 0.2, 0.1]]
            blue_box_left_center = [[0.075, 0.2, 0.03], [0.075, 0.2, 0.03]]
            blue_box_right_back = [[0.175, 0.15, 0.03], [0.175, 0.15, 0.03]]
            blue_box_right_front = [[0.175, 0.25, 0.03], [0.175, 0.25, 0.03]]

            self.goals = np.stack([back_left, back_center, back_right, in_between_boxes, blue_box_left_center, blue_box_right_back, blue_box_right_front])  
        
        elif self.task in ["shelf-place"]:
            front_left = [[-0.15, 0, 0.03], [-0.15, 0, 0.03]]
            back_left = [[-0.3, 0.25, 0.03], [-0.3, 0.25, 0.03]]
            front_right = [[0.15, 0, 0.03], [0.15, 0, 0.03]]
            back_right = [[0.2, 0.25, 0.03], [0.2, 0.25, 0.03]]
            front_center = [[0, 0, 0.03], [0, 0, 0.03]]
            on_shelf = [[-0.1, 0.3, 0.275], [-0.1, 0.3, 0.275]]
            
            self.goals = np.stack([front_left, back_left, front_right, back_right, front_center, on_shelf])
        else:
            NotImplementedError
        
        return self.goals
    
    def get_random_goal(self):
        goals = self.set_goals_for_task()
        return goals[np.random.randint(len(goals))]

    def get_goal(self, index):
        goals = self.set_goals_for_task()
        return goals[index]
    
    def render(self):
        return cv2.resize(self._env.mujoco_renderer.render(render_mode="rgb_array", camera_name=self.camera)[::-1].copy(), self.size, interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
    
    def set_goal_state(self, target_pos):
        pos = target_pos + self.object_start_pos
        self._env._set_obj_xyz(pos)
        # self._env.obj_init_pos = self._env._get_pos_objects() # remove for testing, should be reintegrated
        
        return self.step(np.zeros_like(self.act_space["action"].sample()))
            