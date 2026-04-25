import os
import random

from copy import deepcopy
from typing import Dict, Tuple, Union, List, Any, Optional, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexsim.types import PhysicalAttr, ActorType, DriveType

from embodychain.utils.utility import (
    load_json,
    get_random_real_image,
    set_texture_to_material,
)
from embodychain.utils.logger import log_info, log_warning, log_error
from embodychain.embodylab.embodygym.utils.registration import register_env
from embodychain.embodylab.embodygym.envs.manipulation import ManipulationEnv
from embodychain.embodylab.embodygym.motion_generation.action.arm_action import (
    ArmAction,
)
from embodychain.embodylab.embodygym.motion_generation.motion_imitator import (
    MotionImitator,
)
from embodychain.embodylab.embodysim.end_effector.utility import inv_transform
from embodychain.embodylab.embodygym.utils.misc import (
    resolve_formatted_params,
    resolve_env_params,
    is_pose_flip,
    mul_linear_expand,
    find_function,
    _get_valid_grasp,
    resolve_formatted_string,
    get_offset_pose,
    get_offset_pose_list,
    get_changed_pose,
    axis_str_to_list,
    validation_with_process_from_name,
    is_binocularcam,
    parse_mask_by_uuids,
    project_3d_to_2d,
    is_qpos_flip,
    get_pc_svd,
    apply_svd_transfer_pc,
    add_xy_random_offset,
    expand_pose,
    get_rotated_pose,
    get_rotation_replaced_pose,
)

from embodychain.embodylab.embodygym.structs.configurable_action import (
    ActionBank,
    tag_node,
    tag_edge,
    get_func_tag,
)
from embodychain.toolkits.graspkit.pg_grasp import GraspSelectMethod
from embodychain.data import get_data_path


@register_env("OpenPanPickAndPlaceEnv-v1", max_episode_steps=600)
class OpenPanPickAndPlaceEnv(ManipulationEnv):
    def __init__(
        self,
        num_envs=1,  # 环境数量
        render_backend=False,  # 是否开启渲染
        headless=False,  # 是否无头模式
        enable_rt: bool = False,  # 是否开启实时渲染
        **kwargs,  # 其他参数
    ):
        from rlia.kit.utility import TrajectorySampleMethod
        from rlia.kit.drive_controllers.utility import PathPlanningType

        self.affordance_datas = {}  # 存储物体的抓取和放置位姿
        self.planning_config = {  # 路径规划配置
            "sample_method": TrajectorySampleMethod.QUANTITY,  # 采样方法
            "is_linear": False,  # 是否线性路径
        }
        self.apple_start_pose = np.eye(4)  # 锅开始位姿
        self.apple_stop_pose = np.eye(4)  # 锅结束位姿
        self.apple_grasp_pose_object = np.eye(4)  # 锅抓取位姿
        self.apple_place_pose = np.eye(4)  # 锅放置位姿
        self.lid_grasp_pose_object = np.eye(4)  # 锅盖抓取位姿
        self.lid_place_pose = np.eye(4)  # 锅盖放置位姿
        self.b_fail = 0  # 失败计数
        self.c_fail = 0  # 失败计数

        # Legacy attributes for compatibility with old bottle/cup code
        self.apple_approach_direction = np.array([0, 0, -1])
        self.apple_grasp_xpos_with_offset = np.eye(4)
        self.lib_grasp_xpos_under_grasp_sys = np.eye(4)
        self.lib_opening = np.eye(4)

        self.camera_dict = {}  # 存储相机的初始位姿和内参
        self.dataset_uuid_dict = {}  # 存储需要保存数据集的物体uuid
        self.flag = False
        self.total_nums = 0  # 总计数
        self.supported_robots += ["CobotMagic", "DexAloha"]  # 支持的机械臂类型

        super().__init__(
            num_envs=num_envs,
            headless=headless,
            render_backend=render_backend,
            enable_rt=enable_rt,
            **kwargs,
        )

        self.strict = self.metadata["success_params"].get("strict", False)

        self.control_dict = self.metadata["robot_action"].get("control_parts", None)

        self.init_articulation_drive_param()
        grab_vertices, _ = self.agent.articulation.get_link_vert_face("right_link7")
        self.grab_y_max = grab_vertices[:, 1].max()
        self.grab_y_min = grab_vertices[:, 1].min()

        self._default_physical_attr = PhysicalAttr()
        self._default_physical_attr.mass = 0.005
        self._default_physical_attr.static_friction = 2.0
        self._default_physical_attr.dynamic_friction = 1.5
        self._default_physical_attr.linear_damping = 0.9
        self._default_physical_attr.angular_damping = 0.9
        self._default_physical_attr.contact_offset = 0.003
        self._default_physical_attr.rest_offset = 0.001
        self._default_physical_attr.restitution = 0.1
        self._default_physical_attr.max_depenetration_velocity = 1e1
        self._default_physical_attr.max_linear_velocity = 1e0
        self._default_physical_attr.max_angular_velocity = 1e0

        self.table = self.scene.get_fixed_actor(
            self.scene.get_fixed_actor_uid_list()[0]
        )
        self.table.set_physical_attr(self._default_physical_attr)
        self.table_default_position = self.table.get_location()

        self.agent_type_name = self.agent.__class__.__name__
        from embodychain.embodylab.embodygym.robots.dual_agile import DualAgile

        if isinstance(self.agent, DualAgile) is True:
            self.single_action_space = self.agent.get_single_action_space()

        # TODO: to be removed
        if self.agent_type_name == "DualAgile":
            self.left_arm_uid = "LeftManipulator"
            self.right_arm_uid = "RightManipulator"
        elif self.agent_type_name == "DexAloha" or self.agent_type_name == "CobotMagic":
            self.left_arm_uid = "left_arm"
            self.right_arm_uid = "right_arm"
            self.agent_qpos_flip_ids = [3, 4]
            self.agent_qpos_flip_threshold = 1.1 * np.pi
            self.agent_qpos_flip_mode = "delta"
        elif "DexforceW1" in self.agent_type_name:
            self.left_arm_uid = "left_arm"
            self.right_arm_uid = "right_arm"
            self.agent_qpos_flip_ids = [3]
            self.agent_qpos_flip_threshold = None
            self.agent_qpos_flip_mode = "sign"
        self._init_remember_uuids_for_datasets()
        self._init_camera_origin(arena_index=-1)
        self.randomization_imgs = get_data_path("CocoBackground/coco")
        articulation = self.get_agent().get_articulation()

        link_names = articulation.get_link_names()
        self.agent_mat = self.scene.create_material("link_mat", "pbr")
        for link_name in link_names:
            mat_inst = self.agent_mat.get_inst(link_name)
            articulation.set_material(link_name, mat_inst)

        # ------------------------------------action back related---------------------------------------------
        self.affordance_datas["lid_grasp_pose_object"] = self.lid_grasp_pose_object
        action_config = kwargs.get("action_config", None)
        if action_config is None:
            log_error(
                f"The action config is None, but it's needed for Env: {type(self).__name__}, Task Type: {self.metadata['task_type']}."
            )
        self.action_bank = OpenPanActionBank(action_config)

        vis_graph = action_config.get("other", {}).get("vis_graph", False)
        self.graph_compose, jobs_data, jobkey2index = self.action_bank.parse_network(
            get_func_tag("node").functions[self.action_bank.__class__.__name__],
            get_func_tag("edge").functions[self.action_bank.__class__.__name__],
            vis_graph=vis_graph,
        )

        vis_gantt = action_config.get("other", {}).get("vis_gantt", False)
        self.packages = self.action_bank.gantt(jobs_data, jobkey2index, vis=vis_gantt)

    def init_articulation_drive_param(self):
        left_joint_ids = self.agent.get_joint_ids("left_eef")
        right_joint_ids = self.agent.get_joint_ids("right_eef")
        self.agent.articulation.set_drive(
            stiffness=1e3,
            damping=1e2,
            max_force=1e4,
            drive_type=DriveType.FORCE,
            joint_ids=np.hstack([left_joint_ids, right_joint_ids]),
        )

    def _init_camera_origin(self, arena_index):
        for name, sensor in self._sensors.items():
            sensor = self.get_sensor(name=name, arena_index=arena_index)
            if name in self.camera_dict.keys():
                log_info("Skip camera {} xpos setting.".format(name))
                continue
            pose = deepcopy(sensor.get_world_pose())
            self.camera_dict[name] = {}
            self.camera_dict[name]["pose"] = pose
            if type(sensor).__name__ == "BinocularCam":
                intrinsic = sensor._monocular_cam_l.get_intrinsic()
            elif type(sensor).__name__ == "MonocularCam":
                intrinsic = sensor.get_intrinsic()
            self.camera_dict[name]["intrinsic"] = intrinsic

    def _init_remember_uuids_for_datasets(self, **kwargs):
        robot_uuids = self.agent.get_articulation(self.agent.uid).get_user_ids()

        table_uuid = self.table.get_user_id()

        self.dataset_uuid_dict["robot"] = robot_uuids
        self.dataset_uuid_dict["table"] = table_uuid

    def _init_sim_state(self, **kwargs):
        # TODO: for setup motion imitator.
        # self._robotic_actions = MotionImitator(self, **self.metadata["robot_action"])
        # TODO: special selection for objects.
        self.pan_object_candidates = [
            obj for obj in self.metadata["objects"] if obj.name == "pan"
        ]
        self.lid_object_candidates = [
            obj for obj in self.metadata["objects"] if obj.name == "lid"
        ]
        self.apple_candidates = [
            obj for obj in self.metadata["objects"] if obj.name == "apple"
        ]

        # initialize domain randomization factors.
        self._setup_domain_randomization_settings()

        # TODO: make material dr configurable.
        if self.is_do_material_randomization():
            # initialize table materials.
            from embodychain.data import DexsimMaterials

            table_mat_data = DexsimMaterials()
            table_file_list = os.listdir(table_mat_data.extract_dir)
            table_file_list = [
                os.path.join(table_mat_data.extract_dir, mat) for mat in table_file_list
            ]
            self.table_mat_list = []
            for i, file in enumerate(table_file_list):
                mat = self.scene.get_env().load_material(file, f"table_mat_{i}")
                self.table_mat_list.append(mat)

            # initialize plastic materials.
            from embodychain.data import DefaultPlasticMaterials

            plastic_mat_data = DefaultPlasticMaterials()
            plastic_file_list = os.listdir(plastic_mat_data.extract_dir)
            plastic_file_list = [
                os.path.join(plastic_mat_data.extract_dir, mat)
                for mat in plastic_file_list
            ]
            self.plastic_mat_list = []
            for i, file in enumerate(plastic_file_list):
                mat = self.scene.get_env().load_material(file, f"plastic_mat_{i}")
                self.plastic_mat_list.append(mat)

        self.recover_scale = 1.0

    def _initialize_episode(self, arena_index: int = -1, **kwargs):
        for item in self.scene.get_dynamic_actor_uid_list():
            self.scene.remove_dynamic_actor(item)

        table_position = deepcopy(self.table_default_position)
        shift_value = self.metadata["robot_action"].get("table_z_shift", 0.0)

        z_shift = np.random.uniform(low=-shift_value, high=shift_value)
        self.z_shift = z_shift
        table_position[2] += z_shift
        self.table.set_location(table_position[0], table_position[1], table_position[2])

        super()._initialize_episode(arena_index, **kwargs)

        # TODO: should implemented in manipulation env so that inherited env can enjoy some natural domain
        # randomization, e.g. randomizing init xpos of robot, randomizing init camera xpos and intrinsic parameters.
        self._randomize_init_xpos()
        self._randomize_camera_xpos(arena_index)
        self._randomize_camera_intrinsic(arena_index)

        # TODO:
        self.table.set_body_scale(1.0, self.recover_scale, self.recover_scale)
        table_xy_scale = self.metadata["robot_action"].get("table_xy_scale", [1.0, 1.0])
        scale = np.random.uniform(low=table_xy_scale[0], high=table_xy_scale[1])
        self.table.set_body_scale(1.0, scale, scale)
        self.recover_scale = 1.0 / scale

    def _randomize_camera_xpos(self, arena_index):
        # apply random camera pose.
        camera_xpos_range = self.metadata["robot_action"].get("camera_xpos_range", None)

        if camera_xpos_range is None:
            log_warning("No valid camera xpos range.")
            return

        for name, sensor in self._sensors.items():
            sensor = self.get_sensor(name=name, arena_index=arena_index)
            pose = deepcopy(self.camera_dict[name]["pose"])
            if name in camera_xpos_range.keys():
                if "position" in camera_xpos_range[name].keys():
                    xyz_shift = np.random.uniform(
                        low=camera_xpos_range[name]["position"][0],
                        high=camera_xpos_range[name]["position"][1],
                    )
                    pose[:3, 3] += xyz_shift
                if "rotation" in camera_xpos_range[name].keys():
                    rind = np.random.choice([0, 1, 2])
                    axis = np.zeros((3))
                    axis[rind] = np.deg2rad(
                        np.random.uniform(
                            low=camera_xpos_range[name]["rotation"][0][rind],
                            high=camera_xpos_range[name]["rotation"][1][rind],
                        )
                    )
                    t = np.eye(4)
                    t[:3, :3] = R.from_rotvec(axis).as_matrix()
                    pose = np.matmul(pose, t)
                # TODO: improve sensor coordinate system interface.
                if name == "cam_high":
                    sensor.set_world_pose(pose)
                else:
                    sensor._camera.set_local_pose(pose)

    def _randomize_camera_intrinsic(self, arena_index):
        # apply random camera pose.
        camera_intrinsic_range = self.metadata["robot_action"].get(
            "camera_intrinsic_range", None
        )

        if camera_intrinsic_range is None:
            log_warning("No valid camera intrinsic range.")
            return

        for name, sensor in self._sensors.items():
            sensor = self.get_sensor(name=name, arena_index=arena_index)
            intrinsic = deepcopy(self.camera_dict[name]["intrinsic"])
            if name in camera_intrinsic_range.keys():
                if "focal" in camera_intrinsic_range[name].keys():
                    assert len(camera_intrinsic_range[name]["focal"][0]) == 2
                    assert len(camera_intrinsic_range[name]["focal"][1]) == 2
                    focal_shift = np.random.uniform(
                        low=camera_intrinsic_range[name]["focal"][0],
                        high=camera_intrinsic_range[name]["focal"][1],
                    )
                    intrinsic[0, 0] += focal_shift[0]
                    intrinsic[1, 1] += focal_shift[1]

                if type(sensor).__name__ == "BinocularCam":
                    sensor._monocular_cam_l.set_intrinsic(intrinsic)
                    sensor._monocular_cam_r.set_intrinsic(intrinsic)
                elif type(sensor).__name__ == "MonocularCam":
                    sensor.set_intrinsic(intrinsic)

    def _randomize_init_xpos(
        self,
    ):
        # apply random init state.
        init_xpos_range_config = self.metadata["robot_action"].get(
            "init_xpos_range", None
        )
        if init_xpos_range_config is None:
            return

        agent = self.get_agent()
        init_xpos_left = agent.get_current_xpos(self.left_arm_uid)
        init_xpos_right = agent.get_current_xpos(self.right_arm_uid)

        current_qpos = agent.get_current_qpos(agent.uid)
        left_arm_indices = agent.get_joint_ids(self.left_arm_uid)
        right_arm_indices = agent.get_joint_ids(self.right_arm_uid)

        xyz_shift_left = np.random.uniform(
            low=init_xpos_range_config["position"][0],
            high=init_xpos_range_config["position"][1],
        )

        init_xpos_left[:3, 3] += xyz_shift_left
        ret, qpos_left = self._get_arm_ik(init_xpos_left, is_left=True)
        if ret and is_qpos_flip(
            qpos_left,
            current_qpos[left_arm_indices],
            qpos_ids=self.agent_qpos_flip_ids,
            threshold=self.agent_qpos_flip_threshold,
            mode=self.agent_qpos_flip_mode,
            return_inverse=True,
        ):
            current_qpos[left_arm_indices] = qpos_left
        else:
            log_warning(f"Init left arm qpos failed.\n")

        xyz_shift_right = np.random.uniform(
            low=init_xpos_range_config["position"][0],
            high=init_xpos_range_config["position"][1],
        )

        init_xpos_right[:3, 3] += xyz_shift_right
        ret, qpos_right = self._get_arm_ik(init_xpos_right, is_left=False)
        if ret and is_qpos_flip(
            qpos_right,
            current_qpos[right_arm_indices],
            qpos_ids=self.agent_qpos_flip_ids,
            threshold=self.agent_qpos_flip_threshold,
            mode=self.agent_qpos_flip_mode,
            return_inverse=True,
        ):
            current_qpos[right_arm_indices] = qpos_right
        else:
            log_warning(f"Init right arm qpos failed.\n")

        agent.set_current_qpos(agent.uid, current_qpos)
        self.scene.update(step=100)

    def clean_final(
        self,
    ):
        self.scene.clean_material_cache()
        self.scene.set_default_background("dark")
        articulation = self.get_agent().get_articulation()
        self.agent_mat = {}

        link_names = articulation.get_link_names()
        for link_name in link_names:
            mat = self.scene.create_material(link_name, "pbr")
            self.get_agent().articulation.set_material(link_name, mat.get_material())
            self.agent_mat[link_name] = mat

        if self.is_do_material_randomization():
            # initialize table materials.
            from embodychain.data import DexsimMaterials

            table_mat_data = DexsimMaterials()
            table_file_list = os.listdir(table_mat_data.extract_dir)
            table_file_list = [
                os.path.join(table_mat_data.extract_dir, mat) for mat in table_file_list
            ]
            self.table_mat_list = []
            for i, file in enumerate(table_file_list):
                mat = self.scene.get_env().load_material(file, f"table_mat_{i}")
                self.table_mat_list.append(mat)

            # initialize plastic materials.
            from embodychain.data import DefaultPlasticMaterials

            plastic_mat_data = DefaultPlasticMaterials()
            plastic_file_list = os.listdir(plastic_mat_data.extract_dir)
            plastic_file_list = [
                os.path.join(plastic_mat_data.extract_dir, mat)
                for mat in plastic_file_list
            ]
            self.plastic_mat_list = []
            for i, file in enumerate(plastic_file_list):
                mat = self.scene.get_env().load_material(file, f"plastic_mat_{i}")
                self.plastic_mat_list.append(mat)

    def _update_sim_state(self, arena_index: int = -1, **kwargs):
        super()._update_sim_state(arena_index, **kwargs)

        if self.is_do_domain_randomization() and self.is_do_material_randomization():
            plane_material = self.scene.get_material("plane_mat")
            plane_material.set_base_color_map(
                get_random_real_image(self.randomization_imgs, read=False)
            )

            table_mat = np.random.choice(
                self.table_mat_list, size=1, replace=False
            ).tolist()[0]

            pan_mat = np.random.choice(
                self.plastic_mat_list, size=1, replace=False
            ).tolist()[0]
            lid_mat = np.random.choice(
                self.plastic_mat_list, size=1, replace=False
            ).tolist()[0]
            apple_mat = np.random.choice(
                self.plastic_mat_list, size=1, replace=False
            ).tolist()[0]

            def random_texture_or_uniform_color(mat, inst_name=""):
                if np.random.rand() > 0.5:
                    mat.set_base_color_map(
                        get_random_real_image(self.randomization_imgs, read=False),
                        inst_name,
                    )

                else:
                    draw_color = 255.0 * np.ones(shape=(100, 100, 3), dtype=np.uint8)
                    draw_color *= np.random.rand(1, 1, 3)
                    draw_color = draw_color.astype(np.uint8)
                    set_texture_to_material(mat, draw_color, self.scene.get_env())

            random_texture_or_uniform_color(table_mat)
            random_texture_or_uniform_color(pan_mat)
            random_texture_or_uniform_color(lid_mat)
            random_texture_or_uniform_color(apple_mat)

            self.table.set_material(table_mat)
            self.pan.set_material(pan_mat)
            self.lid.set_material(lid_mat)
            self.apple.set_material(apple_mat)

            articulation = self.get_agent().get_articulation()

            link_names = articulation.get_link_names()
            for link_name in link_names:
                random_texture_or_uniform_color(self.agent_mat, link_name)

    def get_simulated_trajectory_num(self):
        return 1

    ####################################### Analytic ######################################

    def _get_arm_ik(
        self,
        target_xpos: np.ndarray,
        is_left: bool = True,
        qpos_seed: np.ndarray = None,
    ):
        agent = self.get_agent()
        if is_left:
            uid = self.left_arm_uid
        else:
            uid = self.right_arm_uid
        if qpos_seed is None:
            joint_ids = agent.get_joint_ids(name=uid)
            qpos_seed = agent.get_init_qpos(agent.uid)[joint_ids]
        if self.agent_type_name == "DualAgile":
            base_pose = agent.get_base_xpos(name=uid)
            return agent.get_ik(
                xpos=inv_transform(base_pose) @ target_xpos,
                uid=uid,
                qpos_seed=qpos_seed,
                is_world_coordinates=False,
            )
        elif self.agent_type_name == "DexAloha" or self.agent_type_name == "CobotMagic":
            return agent.get_ik(
                xpos=target_xpos,
                uid=uid,
                qpos_seed=qpos_seed,
                is_world_coordinates=True,
            )
        elif "DexforceW1" in self.agent_type_name:
            return agent.get_ik(
                xpos=target_xpos,
                uid=uid,
                qpos_seed=qpos_seed,
                is_world_coordinates=True,
            )
        else:
            log_error("Unsupported robot type.")

    def _transfer_grasp_pose_to_world(
        self, grasp_pose_obj: np.ndarray, obj_pose: np.ndarray
    ):
        """Transfer the grasp pose to world frame.
           And for CobotMagic only, rotate the grasp pose under world frame around z axis for pi / 2, as the gripper axis is different.

        Args:
            grasp_pose_obj (np.ndarray): Grasp pose under object system
            obj_pose (np.ndarray): _description_
        """
        # Transform grasp pose from object frame to world frame
        grasp_world = obj_pose @ grasp_pose_obj  # T_w_g = T_w_o @ T_o_g

        if self.agent.uid == "CobotMagic":
            # Exchange x and y axes to adjust gripper orientation
            # New x = -old y
            # New y = old x
            grasp_rx = deepcopy(grasp_world[:3, 0])
            grasp_world[:3, 0] = -grasp_world[:3, 1]
            grasp_world[:3, 1] = grasp_rx

        return grasp_world

    def _get_grasp_pose_list(
        self,
        grasp_pose_obj: np.ndarray,
        obj_pose: np.ndarray,
        approach_direction: np.ndarray = None,
    ):
        r"""Calculate grasp poses list for robotic manipulation

        TODO: The process of pre1 and pre2 should be done in env.affordance_data, not this function

        Args:
            grasp_pose_obj: Grasp pose in object coordinate frame
            obj_pose: Object pose in world coordinate frame
            approach_direction: Backwards direction in world coordinate frame
        Returns:
            grasp_world: Final grasp pose in world frame
            pre1_pose_world: Pre-grasp pose 1 in world frame (closer to grasp)
            pre2_pose_world: Pre-grasp pose 2 in world frame (further from grasp)
        """

        grasp_world = self._transfer_grasp_pose_to_world(grasp_pose_obj, obj_pose)

        # Extract approach direction (z-axis of grasp pose)
        grasp_z = grasp_world[:3, 2]

        # Ensure z-axis points downward for stable grasping
        # grasp_world = align_pose_with_z_down(grasp_world)

        # Generate pre-grasp poses with different offsets based on robot type
        if "DexforceW1" in self.agent_type_name:
            # Create pre-grasp pose 1 (5cm offset)
            pre1_pose_world = deepcopy(grasp_world)
            offset_matrix1 = np.eye(4)
            offset_matrix1[:3, 3] = 0.05 * grasp_z
            pre1_pose_world = pre1_pose_world @ offset_matrix1

            # Create pre-grasp pose 2 (10cm offset)
            pre2_pose_world = deepcopy(grasp_world)
            offset_matrix2 = np.eye(4)
            offset_matrix2[:3, 3] = 0.1 * grasp_z
            pre2_pose_world = pre2_pose_world @ offset_matrix2
        else:
            if approach_direction is None:
                approach_direction = self.apple_approach_direction
            pre1_pose_world = deepcopy(grasp_world)
            pre1_pose_world[0:3, 3] -= 0.05 * approach_direction
            pre2_pose_world = deepcopy(grasp_world)
            pre2_pose_world[0:3, 3] -= 0.1 * approach_direction

        return grasp_world, pre1_pose_world, pre2_pose_world

    def test_grasp_pose_reachability(self, object_type="apple"):
        """Test if the grasp pose is reachable by the corresponding arm

        Args:
            object_type (str): "apple" for right arm or "lid" for left arm
        """
        # print(f"=== Testing {object_type.capitalize()} Grasp Pose Reachability ===")

        # Get current robot state
        init_qpos = self.agent.get_current_qpos(name=self.agent.uid)

        if object_type == "apple":
            # Right arm for apple
            arm_joints = self.agent.get_joint_ids(name=self.right_arm_uid)
            arm_init_qpos = init_qpos[arm_joints]
            is_left = False
            target_object = self.apple
            grasp_pose_object = self.apple_grasp_pose_object
            grasp_offset = self.apple_grasp_offset
        else:  # lid
            # Left arm for lid
            arm_joints = self.agent.get_joint_ids(name=self.left_arm_uid)
            arm_init_qpos = init_qpos[arm_joints]
            is_left = True
            target_object = self.lid
            grasp_pose_object = self.lid_grasp_pose_object
            grasp_offset = self.lid_grasp_offset

        # Calculate grasp pose in world coordinates
        grasp_world = target_object.get_world_pose() @ grasp_pose_object
        grasp_offset_matrix = np.eye(4)
        grasp_offset_matrix[2, 3] = grasp_offset
        grasp_world = grasp_world @ grasp_offset_matrix

        # print(f"{object_type.capitalize()} world pose:\n{target_object.get_world_pose()}")
        # print(f"{object_type.capitalize()} grasp pose (object frame):\n{grasp_pose_object}")
        # print(f"{object_type.capitalize()} grasp offset: {grasp_offset}")
        # print(f"{object_type.capitalize()} grasp pose (world frame):\n{grasp_world}")

        # Test IK for grasp pose
        ret, grasp_qpos = self._get_arm_ik(
            grasp_world,
            is_left=is_left,
            qpos_seed=arm_init_qpos,
        )

        # print(f"IK result: {ret}")
        if ret:
            # print(f"Grasp qpos: {grasp_qpos}")

            # Test for qpos flip
            flip_result = is_qpos_flip(
                grasp_qpos,
                arm_init_qpos,
                self.agent_qpos_flip_ids,
                threshold=self.agent_qpos_flip_threshold,
                mode=self.agent_qpos_flip_mode,
            )
            # print(f"Qpos flip check: {flip_result}")

            if not flip_result:
                # print(f"✓ {object_type.capitalize()} grasp pose is REACHABLE and valid!")

                # Verify by forward kinematics
                arm_name = "left_arm" if is_left else "right_arm"
                fk_result = self.agent.get_fk(
                    grasp_qpos, arm_name, is_world_coordinates=True
                )
                # print(f"Forward kinematics verification:\n{fk_result}")

                # Compare with target
                position_error = np.linalg.norm(fk_result[:3, 3] - grasp_world[:3, 3])
                # print(f"Position error: {position_error:.6f} meters")

                return True
            else:
                # print(f"✗ {object_type.capitalize()} grasp qpos has flip issue")
                return False
        else:
            # print(f"✗ {object_type.capitalize()} grasp pose is NOT reachable")
            return False

    def _setup_simulated_objects(self, **kwargs):
        # Setup pan (锅)
        pan = random.choice(self.pan_object_candidates)
        pan_xpos = add_xy_random_offset(
            np.array(pan.pose),
            self.metadata["robot_action"]["max_random_shift"]["pan"],
        )

        # Setup lid (锅盖) - use same random offset as pan to ensure lid covers pan
        lid = random.choice(self.lid_object_candidates)
        self.lid_xy_random_center = np.asarray(pan.pose)[:2, 3] + [
            sum(self.metadata["robot_action"]["max_random_shift"]["pan"][:2]) / 2,
            sum(self.metadata["robot_action"]["max_random_shift"]["pan"][2:]) / 2,
        ]

        lid_base_pose = np.array(lid.pose)
        # Apply the same offset that was applied to pan
        pan_offset = pan_xpos[:2, 3] - np.array(pan.pose)[:2, 3]
        lid_xpos = lid_base_pose.copy()
        lid_xpos[:2, 3] += pan_offset

        # Setup apple to place (要放置的物体)
        apple = random.choice(self.apple_candidates)
        self.apple_xy_random_center = np.asarray(apple.pose)[:2, 3] + [
            sum(self.metadata["robot_action"]["max_random_shift"]["apple"][:2]) / 2,
            sum(self.metadata["robot_action"]["max_random_shift"]["apple"][2:]) / 2,
        ]
        apple_xpos = add_xy_random_offset(
            np.array(apple.pose),
            self.metadata["robot_action"]["max_random_shift"]["apple"],
        )
        if "carrot" in apple.get_mesh_file():
            rotation_range = self.metadata["robot_action"].get(
                "rotation_range", [[0, 0], [0, 0], [-45, 45]]
            )
            rx_range = rotation_range[0]
            ry_range = rotation_range[1]
            rz_range = rotation_range[2]
            rot = R.from_euler(
                "xyz",
                [
                    np.random.uniform(rx_range[0], rx_range[1]),
                    np.random.uniform(ry_range[0], ry_range[1]),
                    np.random.uniform(rz_range[0], rz_range[1]),
                ],
                degrees=True,
            )
            apple_xpos[:3, :3] = rot.as_matrix() @ apple_xpos[:3, :3]
        # Calculate apple dimensions for manipulation

        self.apple_grasp_pose_object = apple.grasp_pose_object
        self.apple_grasp_offset = apple.grasp_offset

        # Setup lid (锅盖)
        standard_lid_verts = apply_svd_transfer_pc(lid)
        self.lid_height = (
            standard_lid_verts[:, 2].max() - standard_lid_verts[:, 2].min()
        )
        self.lid_radius = (
            abs(standard_lid_verts[:, 1].max() - standard_lid_verts[:, 1].min()) / 2
        )

        standard_pan_verts = apply_svd_transfer_pc(pan)
        self.pan_height = (
            standard_pan_verts[:, 2].max() - standard_pan_verts[:, 2].min()
        )
        self.pan_radius = (
            abs(standard_pan_verts[:, 1].max() - standard_pan_verts[:, 1].min()) / 2
        )

        self.lid_grasp_pose_object = lid.grasp_pose_object
        self.lid_grasp_offset = lid.grasp_offset
        # Set control parts - left arm controls lid, right arm controls object
        self.lid_grasp_control_part = "left_arm"
        self.apple_grasp_control_part = "right_arm"

        # Add pan to scene
        self.pan = self.scene.add_dynamic_actor(
            pan.get_mesh_file(),
            pan_xpos,
            pan.scale,
            compute_uv=True,
            is_convex_decomposition=True,
            max_convex_hull_num=64,
        )
        self.pan.set_physical_attr(self._default_physical_attr)

        # Add lid to scene
        self.lid = self.scene.add_dynamic_actor(
            lid.get_mesh_file(),
            lid_xpos,
            lid.scale,
            compute_uv=True,
            is_convex_decomposition=True,
            max_convex_hull_num=64,
        )
        self.lid.set_physical_attr(self._default_physical_attr)

        # Add apple to scene
        self.apple = self.scene.add_dynamic_actor(
            apple.get_mesh_file(),
            apple_xpos,
            apple.scale,
            compute_uv=True,
            is_convex_decomposition=True,
            max_convex_hull_num=8,
        )
        self.apple.set_physical_attr(self._default_physical_attr)

        # Apply material randomization if enabled
        if self.is_do_domain_randomization() and self.is_do_material_randomization():
            self.pan.set_material(
                np.random.choice(self.plastic_mat_list, size=1, replace=False).tolist()[
                    0
                ]
            )
            self.lid.set_material(
                np.random.choice(self.plastic_mat_list, size=1, replace=False).tolist()[
                    0
                ]
            )
            self.apple.set_material(
                np.random.choice(self.plastic_mat_list, size=1, replace=False).tolist()[
                    0
                ]
            )

        self.scene.update(step=100)
        # NOTE: fix the pan

        self.pan.set_actor_type(ActorType.KINEMATIC)

        # Test grasp pose reachability
        self.test_grasp_pose_reachability()

        self.lid_max_z = (
            standard_lid_verts[:, 0].max() + self.lid.get_world_pose()[2, 3]
        )

        self.left_arm_base_xy = self.agent.articulation.get_link_pose("left_base_link")[
            :2, 3
        ]
        self.right_arm_base_xy = self.agent.articulation.get_link_pose(
            "right_base_link"
        )[:2, 3]
        self.right_eef_xyz = self.agent.get_current_xpos("right_arm")[
            :3, 3
        ]  # End-effector position
        self.right_eef_rz = self.agent.get_current_xpos("right_arm")[
            :3, 2
        ]  # End-effector z-axis direction
        self_right_ori_rotation = self.agent.get_current_xpos("right_arm")[:3, :3]
        # Set up apple orientation for right arm manipulation
        self.apple_pose_orig = self.apple.get_world_pose()
        self.apple_place_rotate = np.array(apple.place_object_rotation)
        self.right_aim_horizontal_angle = 0.0  # No rotation
        self.apple_rotated_pose = deepcopy(self.apple_pose_orig)  # Keep original pose
        # No rotation applied - only xy translation is used

        # Set up lid orientation for left arm manipulation
        self.lid_pose_orig = self.lid.get_world_pose()
        self.left_aim_horizontal_angle = 0.0  # No rotation
        self.lid_rotated_pose = deepcopy(self.lid_pose_orig)  # Keep original pose
        # No rotation applied - only xy translation is used
        self.apple_pose_orig = self.apple.get_world_pose()
        vertical_down_rotation = R.from_euler("z", -160, degrees=True).as_matrix()
        grasp_apple_rotation = (
            inv_transform(self.apple_pose_orig)[:3, :3] @ vertical_down_rotation
        )
        # print("apple grasp rotation:\n", grasp_apple_rotation)

    #################################### ALL ####################################

    def create_demo_action_list(
        self,
        is_grasp_pose_visual: bool = False,
        **kwargs,
    ):
        try:
            actions = self.create_demo_action_list_analytic(
                is_grasp_pose_visual, **kwargs
            )
        except Exception as e:
            import traceback

            tb_lines = traceback.format_exc().splitlines()
            if len(tb_lines) >= 3:
                indent_len = len(tb_lines[-3]) - len(tb_lines[-3].lstrip())
                traceback_file_path = tb_lines[-3].lstrip()
                traceback_content = tb_lines[-2][indent_len:]
                error = tb_lines[-1]
            else:
                error = tb_lines[-1] if tb_lines else str(e)
            log_warning(traceback_file_path)
            log_warning(traceback_content)
            log_warning(error)
            actions = None
        return actions

    def _extend_obs(
        self, obs: Dict[str, any], arena_index: int = -1, **kwargs
    ) -> Dict[str, any]:
        if "exteroception" in self.metadata["dataset"]["robot_meta"][
            "observation"
        ] and hasattr(self, "left_arm_uid"):
            self._extend_exteroception(obs, arena_index, **kwargs)
        return obs

    def _extend_exteroception(
        self,
        obs: Dict[str, any],
        arena_index: int = -1,
        goal_pose_string: Dict = {},
        **kwargs,
    ) -> Dict[str, any]:
        cams = self.metadata["dataset"]["robot_meta"]["observation"]["exteroception"][
            "cameras"
        ]
        # cam_left_wrist, cam_right_wrist, left_arm, right_arm
        obs["exteroception"] = {}
        main_cam = self.get_sensor("cam_high")
        # TODO: enable genral exteroception compution for robot learning.
        if not is_binocularcam(main_cam):
            return obs

        # bottle is grasped in the trajecory, so exteroception is valid
        if hasattr(self, "left_arm_uid"):
            grab_ratios = {}
            for executor_group in ["left", "right"]:
                eef_executor_name = executor_group + "_eef"
                goal_executor_name = executor_group + "_arm"
                right_grab_ratio = (
                    obs["agent"][eef_executor_name]["qpos"][0]
                    / self.agent.get_joint_limits(eef_executor_name)[0, -1]
                )
                left_grab_ratio = (
                    obs["agent"][eef_executor_name]["qpos"][1]
                    / self.agent.get_joint_limits(eef_executor_name)[1, -1]
                )
                grab_ratios[goal_executor_name] = [left_grab_ratio, right_grab_ratio]

            exteroception_dict = self._def_exteroception(grab_ratios, goal_pose_string)

            if "cam_left_wrist" in cams and "cam_right_wrist" in cams:
                keypoints = self._assign_grasps_to_cams(exteroception_dict)
            else:
                log_warning("No left and right wrist camera.\n")
                keypoints = {}

            if "cam_high" in cams:
                keypoints_in_main_l = project_3d_to_2d(
                    main_cam._monocular_cam_l, exteroception_dict["cam_high"]
                )
                keypoints_in_main_r = project_3d_to_2d(
                    main_cam._monocular_cam_r, exteroception_dict["cam_high"]
                )
                keypoints.update(
                    {"cam_high": {"l": keypoints_in_main_l, "r": keypoints_in_main_r}}
                )
            else:
                log_warning("No high camera.\n")

            obs["exteroception"] = keypoints
            return obs
        else:
            return obs

    def _def_exteroception(
        self, grab_ratios: Dict[str, List], goal_pose_sting: Dict = {}
    ) -> Dict[str, List[np.ndarray]]:
        cams = self.metadata["dataset"]["robot_meta"]["observation"]["exteroception"][
            "cameras"
        ]
        groups = self.metadata["dataset"]["robot_meta"]["observation"][
            "exteroception"
        ].get("groups", 6)
        kpnts_number = self.metadata["dataset"]["robot_meta"]["observation"][
            "exteroception"
        ].get("kpnts_number", 12)
        x_interval, y_interval = self.metadata["dataset"]["robot_meta"]["observation"][
            "exteroception"
        ]["interval"]

        agent = self.get_agent()

        left_arm_exp_offset_matrix = self.affordance_datas.get(
            "lid_grasp_unoffset_matrix_object", np.eye(4)  # TODO
        )
        right_arm_exp_offset_matrix = self.affordance_datas.get(
            "apple_grasp_unoffset_matrix_object", np.eye(4)
        )

        left_tcp_pose_unoffset_world = (
            agent.get_current_xpos(self.left_arm_uid) @ left_arm_exp_offset_matrix
        )
        right_tcp_pose_unoffset_world = (
            agent.get_current_xpos(self.right_arm_uid) @ right_arm_exp_offset_matrix
        )

        grab_ratios["left_arm"] = grab_ratios["left_arm"]
        grab_ratios["right_arm"] = grab_ratios["right_arm"]

        left_tcp_exteroception = expand_pose(
            left_tcp_pose_unoffset_world,
            x_interval,
            y_interval,
            kpnts_number,
            grab_ratios["left_arm"],
        )
        right_tcp_exteroception = expand_pose(
            right_tcp_pose_unoffset_world,
            x_interval,
            y_interval,
            kpnts_number,
            grab_ratios["right_arm"],
        )

        goal_pose = {}
        goal_exteroception = {}
        for executor_name, goal_pose_string_i in goal_pose_sting.items():
            goal_pose_i = self.affordance_datas.get(goal_pose_string_i, np.eye(4))
            goal_pose.update({executor_name: goal_pose_i})
            goal_exteroception_i = expand_pose(
                goal_pose_i,
                x_interval,
                y_interval,
                kpnts_number,
                grab_ratios[executor_name],
            )
            goal_exteroception.update({executor_name: goal_exteroception_i})

        default_expanded_poses = expand_pose(
            np.eye(4), x_interval, y_interval, kpnts_number, grab_ratios=None
        )
        left_goal_exteroception = goal_exteroception.get(
            "left_arm", default_expanded_poses
        )
        right_goal_exteroception = goal_exteroception.get(
            "right_arm", default_expanded_poses
        )

        ret = {}
        for cam in cams:
            if cam == "cam_high":
                ret[cam] = (
                    left_tcp_exteroception
                    + right_tcp_exteroception
                    + left_goal_exteroception
                    + right_goal_exteroception
                )
            elif cam == "cam_left_wrist":
                left_cam_exteroception = (
                    left_tcp_exteroception
                    + left_tcp_exteroception
                    + left_goal_exteroception
                    + left_goal_exteroception
                )  # repeat for batch-training.
                ret[cam] = left_cam_exteroception
            elif cam == "cam_right_wrist":
                right_cam_exteroception = (
                    right_tcp_exteroception
                    + right_tcp_exteroception
                    + right_goal_exteroception
                    + right_goal_exteroception
                )  # repeat for batch-training.
                ret[cam] = right_cam_exteroception
            else:
                log_warning(
                    f"Camera '{cam}' is not supported by _def_exteroception() in env '{self.id}'. Skipping.."
                )

            if (len(ret[cam]) / (2 * kpnts_number + 1)) != groups:
                log_error(
                    f"Exteroception groups for {cam} is {(len(ret[cam]) / (2 * kpnts_number + 1))}, other than assigned groups {groups}. Please check again."
                )

        return ret

    def _assign_grasps_to_cams(self, grasps) -> Dict[str, np.ndarray]:
        # no need to refresh
        left_cam = self.get_sensor("cam_left_wrist")
        # no need to refresh
        right_cam = self.get_sensor("cam_right_wrist")

        keypoints = {"cam_left_wrist": None, "cam_right_wrist": None}
        cams = {"cam_left_wrist": left_cam, "cam_right_wrist": right_cam}

        for name, cam in cams.items():
            keypoints[name] = project_3d_to_2d(cam, grasps[name])

        return keypoints

    def _extend_contact_bottle_legal(
        self, obs: Dict[str, any], arena_index: int = -1, **kwargs
    ) -> Dict[str, any]:
        if hasattr(self, "bottle_grasp_control_part"):
            (
                illegal_link,
                illegal_link_names,
                contact_bottle,
            ) = self.examine_contact_obj_link()
            if illegal_link:
                contact_bottle_legal = (
                    False,
                    f"Contact with illegal links {illegal_link_names}.",
                )
                obs["contact_bottle_legal"] = contact_bottle_legal
                return obs
            elif not illegal_link and contact_bottle:
                contact_bottle_pose = self.bottle.get_world_pose()
                contact_bottle_legal = self.examine_contact_bottle_pose(
                    contact_bottle_pose,
                )
                obs["contact_bottle_legal"] = contact_bottle_legal
                return obs

        return obs

    def examine_contact_obj_link(self):
        contact_info = self.agent.articulation.get_leaf_contacts()
        contact_links = contact_info.link_name
        contact_objs = contact_info.nodes

        illegal_link = False
        contact_bottle = False
        illegal_link_names = []
        # if no contact, won't get in this loop!
        for link_name_i, contact_obj in zip(contact_links, contact_objs):
            if link_name_i not in self.legal_contact_links:
                illegal_link = True
                illegal_link_names.append(link_name_i)
            if contact_obj.get_name() == self.bottle_mesh_file_name:
                contact_bottle = True

        return illegal_link, illegal_link_names, contact_bottle

    def examine_contact_bottle_pose(self, contact_bottle_pose, **kwargs):
        right_grab_name = self.legal_contact_links[0]
        left_grab_name = self.legal_contact_links[1]

        # NOTE: grab will move in z under its sys during moving, ao check every time
        right_grab_pose = self.agent.articulation.get_link_pose(right_grab_name)
        left_grab_pose = self.agent.articulation.get_link_pose(left_grab_name)
        left_grab_z_under_right_grab_sys = (
            inv_transform(right_grab_pose) @ left_grab_pose
        )[2, 3]

        contact_bottle_pose_under_right_grab_sys = (
            inv_transform(right_grab_pose) @ contact_bottle_pose
        )
        contact_bottle_y_under_right_grab_sys = (
            contact_bottle_pose_under_right_grab_sys[1, 3]
        )

        contact_bottle_z_under_right_grab_sys = (
            contact_bottle_pose_under_right_grab_sys[2, 3]
        )

        if contact_bottle_z_under_right_grab_sys < 0.0 + 0.7 * self.bottle_radius:
            # z_{bottle} - z_{right_grab}(=0) < \sqrt{2} / 2 * radius
            return (False, "When contact, bottle right to gripper's right grab.")
        elif (
            contact_bottle_z_under_right_grab_sys
            > left_grab_z_under_right_grab_sys - 0.7 * self.bottle_radius
        ):
            # z_{left_grab} - z_{bottle} < \sqrt{2} / 2 * radius
            return (False, "When contact, bottle left to gripper's left grab.")
        elif contact_bottle_y_under_right_grab_sys < self.grab_y_min + 0.005:
            return (False, "When contact, bottle is so behind.")
        elif (
            contact_bottle_y_under_right_grab_sys
            > self.grab_y_max + 0.55 * self.bottle_radius
        ):
            return (False, "When contact, bottle is so in the front.")
        else:
            return (True, "Whne contact, bottle is fit between the grabs.")

    def _extend_pan_pose(
        self, obs: Dict[str, any], arena_index: int = -1, **kwargs
    ) -> Dict[str, any]:
        if hasattr(self, "pan"):
            obs["pan_pose"] = self.pan.get_world_pose()
        return obs

    def _extend_lid_pose(
        self, obs: Dict[str, any], arena_index: int = -1, **kwargs
    ) -> Dict[str, any]:
        if hasattr(self, "lid"):
            obs["lid_pose"] = self.lid.get_world_pose()
        return obs

    def _extend_apple_pose(
        self, obs: Dict[str, any], arena_index: int = -1, **kwargs
    ) -> Dict[str, any]:
        if hasattr(self, "apple"):
            obs["apple_pose"] = self.apple.get_world_pose()
        return obs

    def _get_sensor_obs(self, arena_index, **kwargs) -> Dict[str, any]:
        obs = {}
        if self._disable_sensors_in_step:
            return obs

        for name, sensor in self._sensors.items():
            sensor = self.get_sensor(name=name, arena_index=arena_index)
            sensor.refresh()
            data_dict = {}
            data_dict["rgb"] = sensor.get_rgb_map()

            if is_binocularcam(sensor):
                data_dict["rgb_right"] = sensor.get_right_rgb_map()
                data_dict[
                    "visible_mask_right"
                ] = sensor._monocular_cam_r._camera.get_visible_mask()
                data_dict["depth"] = sensor._monocular_cam_l.get_depth_map()
            else:
                data_dict["depth"] = sensor.get_depth_map()

            data_dict["visible_mask"] = (
                sensor._camera.get_visible_mask()
                if not is_binocularcam(sensor)
                else sensor._monocular_cam_l._camera.get_visible_mask()
            )

            visib_mask_append = {"semantic_mask_l": None}
            if is_binocularcam(sensor):
                visib_mask_append["semantic_mask_r"] = None

            if hasattr(self, "pan") and hasattr(self, "lid") and hasattr(self, "apple"):
                self.dataset_uuid_dict["object"] = [
                    self.pan.get_user_id(),
                    self.lid.get_user_id(),
                    self.apple.get_user_id(),
                ]
            visib_mask_append["semantic_mask_l"] = parse_mask_by_uuids(
                data_dict["visible_mask"], self.dataset_uuid_dict
            )
            if is_binocularcam(sensor):
                visib_mask_append["semantic_mask_r"] = parse_mask_by_uuids(
                    data_dict["visible_mask_right"], self.dataset_uuid_dict
                )
            data_dict.update(visib_mask_append)

            obs[name] = data_dict

        return obs

    def compute_bottle_cup_len_angle(self, bottle_pose, cup_pose):
        bottle_center = bottle_pose[:3, 3]
        cup_center = cup_pose[:3, 3]

        vec = cup_center - bottle_center
        length = np.linalg.norm(vec)
        vec_norm = vec / length

        bottle_z_axis = bottle_pose[:3, 2]
        bottle_z_axis_norm = bottle_z_axis / np.linalg.norm(bottle_z_axis)

        dot_product = np.dot(bottle_z_axis_norm, vec_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        return length, angle

    def is_fall(self, pose: np.ndarray):
        pose_rz = pose[:3, 2]
        world_z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.dot(pose_rz, world_z_axis))
        if angle >= np.pi / 4:
            return True
        else:
            return False

    def is_task_success(self, obs_list: List) -> bool:
        """Judge open pan task success.

        The task is successful if:
        1. Object is placed inside the pan
        2. Lid is placed back on the pan
        3. No objects have fallen
        4. All constraints are satisfied throughout the trajectory

        Args:
            obs_list (List): The observation list collected from the environment conducting the action list.

        Returns:
            bool: If the task is success given the obs_list
        """

        # Check final poses
        pan_final_pose = self.pan.get_world_pose()
        lid_final_pose = self.lid.get_world_pose()
        apple_final_pose = self.apple.get_world_pose()

        # Check if apple is placed inside the pan
        pan_center = pan_final_pose[:2, 3]  # pan x, y center
        apple_center = apple_final_pose[:2, 3]  # apple x, y center
        distance_apple_to_pan = np.linalg.norm(apple_center - pan_center)

        # apple should be close to pan center (within reasonable radius)
        if distance_apple_to_pan > self.pan_radius:  # 10cm tolerance
            log_info(
                f"apple is not placed inside the pan. Distance: {distance_apple_to_pan}"
            )
            return False

        # apple should be at reasonable z (inside pan, not floating)
        pan_z = pan_final_pose[2, 3]
        apple_z = apple_final_pose[2, 3]
        if (
            apple_z < pan_z or apple_z > pan_z + self.pan_height
        ):  # apple should be within pan
            log_info(f"apple z {apple_z} is not appropriate relative to pan z {pan_z}")
            return False

        # Check if lid is placed back on the pan
        lid_center = lid_final_pose[:2, 3]  # lid x, y center
        distance_lid_to_pan = np.linalg.norm(lid_center - pan_center)

        # Lid should be close to pan center
        if distance_lid_to_pan > self.lid_radius / 4:  # 1/ 8 radius tolerance
            log_info(
                f"Lid is not placed back on the pan. Distance: {distance_lid_to_pan}"
            )
            return False

        # Lid should be at appropriate z (above pan and apple)
        lid_z = lid_final_pose[2, 3]
        if (
            lid_z < apple_z or lid_z < pan_z + self.pan_height - 0.02  # 2cm
        ):  # Lid should be above apple, above pan
            log_info(f"Lid z {lid_z} is not appropriate relative to apple z {apple_z}")
            return False

        log_info("Open pan task completed successfully!")
        return True

    def to_dataset(
        self,
        obs_list: list,
        action_list: list,
        id: str,
    ):
        from embodychain.embodylab.data_engine.data_dict_extractor import (
            fetch_imitation_dataset,
        )

        from embodychain.embodylab.embodygym.robots.interface import LearnableRobot

        dataset_path = self.metadata["dataset"].get("save_path", None)
        if dataset_path is None:
            from embodychain.database import database_demo_dir

            dataset_path = database_demo_dir

        # TODO: create imitation dataset folder with name "{task_name}_{robot_type}_{num_episodes}"
        from embodychain.embodylab.embodygym.utils.misc import camel_to_snake

        if self.curr_episode == 0:
            self.folder_name = f"{camel_to_snake(self.__class__.__name__)}_{camel_to_snake(self.agent.__class__.__name__)}"
            if os.path.exists(os.path.join(dataset_path, self.folder_name)):
                self.folder_name = f"{self.folder_name}_{random.randint(0, 1000)}"

        return fetch_imitation_dataset(
            self, obs_list, action_list, id, self.folder_name
        )

    # TODO: This maybe move to ManipulationEnv.
    def map_control_actions_to_env_actions(
        self,
        actions,
        action_type: str = "qpos",
        **kwargs,
    ):
        from embodychain.embodylab.embodygym.robots import LearnableRobot

        action_space_dim = self.single_action_space["qpos"].shape[0]
        agent: LearnableRobot = self.get_agent()

        env_actions = agent.map_control_actions_to_env_actions(
            actions, action_space_dim, action_type
        )
        # set agent specific default qpos of the environment.
        full_index = set(range(action_space_dim))
        control_index_set = set(agent.get_control_index())
        control_index_set = control_index_set | set(
            self.agent.get_joint_ids("left_eef").tolist()
        )
        control_index_set = control_index_set | set(
            self.agent.get_joint_ids("right_eef").tolist()
        )
        static_index = list(full_index - control_index_set)
        env_actions[:, static_index] = np.tile(
            self._agent_home_joint, (len(actions), 1)
        )[:, static_index]
        return [{"qpos": qp} for qp in env_actions]

    def check_truncated(self, obs, info: Dict, arena_index: int = -1) -> bool:
        return False

    def visual_action(self, action_list: list, visual_time: float = 3.0) -> None:
        if action_list is not None:
            action_list[:] = []
        # TODO

    def create_demo_action_list_analytic(self, *args, **kwargs):
        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )

        if ret is None:
            return None

        left_arm_joints = self.agent.get_joint_ids(name=self.left_arm_uid)
        right_arm_joints = self.agent.get_joint_ids(name=self.right_arm_uid)

        # TODO: now all_dim_ref = 14, however all_dim needs 16 to work
        # all_dim_ref = sum([t.shape[0] for t in ret.values()])
        # all_dim = len(self.agent.get_control_index())
        # if all_dim_ref != all_dim:
        #     log_error(f"Got all dim = {all_dim_ref} from ret, but the self.agent itself needs a {all_dim} dim control signal.")
        all_dim = self.agent.get_current_qpos(name=self.agent.uid).shape[-1]

        total_traj_num = ret[list(ret.keys())[0]].shape[-1]

        qpos_new = np.ones((total_traj_num, all_dim), dtype=np.float32)
        qpos_new[:, left_arm_joints] = ret["left_arm"].T
        qpos_new[:, right_arm_joints] = ret["right_arm"].T
        qpos_new = qpos_new.astype(np.float32)

        actions = []
        log_info(f"Total generated trajectory number: {total_traj_num}.", color="green")
        log_info(f"Lid_pos: {self.lid.get_world_pose()[0:3,3]}.", color="green")
        log_info(f"Apple_pos: {self.apple.get_world_pose()[0:3,3]}.", color="green")
        ee_state_list_left = ret["left_eef"].T
        ee_state_list_right = ret["right_eef"].T
        if self.agent_type_name == "DexAloha" or self.agent_type_name == "CobotMagic":
            qpos_new = self.agent.map_ee_state_to_env_actions(
                np.hstack(
                    (np.array(ee_state_list_left), np.array(ee_state_list_right))
                ),
                qpos_new,
            )

        elif "DexforceW1" in self.agent_type_name:
            qpos_new = np.hstack(
                (qpos_new, np.array(ee_state_list_left), np.array(ee_state_list_right))
            )
        else:
            log_error(
                "Unsupported robot type {}. Supported robot type: [`DualAgile`, `DexAloha`, `CobotMagic`]".format(
                    self.agent_type_name
                )
            )
        for i in range(total_traj_num):
            actions.append({"qpos": qpos_new[i]})
        return actions

    def _get_arm_fk(self, qpos, uid, is_world_coordinates=True):
        return self.agent.get_fk(qpos, uid, is_world_coordinates=is_world_coordinates)

    def step(
        self, action: Union[np.ndarray, Dict], **kwargs
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        goal_pose_string = action.pop("goal_pose_string", None)

        if goal_pose_string is not None:
            kwargs.update({"goal_pose_string": goal_pose_string})

        observation, reward, terminated, truncated, info = super().step(
            action, **kwargs
        )

        return observation, reward, terminated, truncated, info


class OpenPanActionBank(ActionBank):
    @staticmethod
    @tag_node
    def generate_left_arm_init_qpos(env: OpenPanPickAndPlaceEnv):
        left_arm_init_pose = env._get_arm_fk(
            env.affordance_datas["left_arm_init_qpos"],
            uid="left_arm",
            is_world_coordinates=True,
        )
        env.affordance_datas["left_arm_init_pose"] = np.array(left_arm_init_pose)
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_grasp(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        lid_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.lid_grasp_pose_object),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_grasp_pose is None:
            return False
        env.affordance_datas["lid_grasp_pose"] = lid_grasp_pose

        lid_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_grasp_qpos is None:
            return False
        env.affordance_datas["lid_grasp_qpos"] = lid_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_pre2_grasp_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        lid_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_pre2_grasp_pose is None:
            return False
        env.affordance_datas["lid_pre2_grasp_pose"] = lid_pre2_grasp_pose

        # Calculate IK for lid pre2 position
        lid_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            env.affordance_datas["lid_pre2_grasp_pose"],
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_pre2_grasp_qpos is None:
            log_warning("Failed to generate lid_pre2_grasp_qpos")
            return False
        env.affordance_datas["lid_pre2_grasp_qpos"] = lid_pre2_grasp_qpos

        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_pre1_grasp_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        lid_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_pre1_grasp_pose is None:
            return False
        env.affordance_datas["lid_pre1_grasp_pose"] = lid_pre1_grasp_pose

        # Calculate IK for lid pre1 position
        lid_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            lid_pre1_grasp_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_pre1_grasp_qpos is None:
            log_warning("Failed to generate lid_pre1_grasp_qpos")
            return False
        env.affordance_datas["lid_pre1_grasp_qpos"] = lid_pre1_grasp_qpos

        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_lift_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        lid_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_lift_pose is None:
            return False
        env.affordance_datas["lid_lift_pose"] = lid_lift_pose

        lid_lift_qpos = validation_with_process_from_name(
            env,
            env.affordance_datas["lid_lift_pose"],
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_lift_qpos is None:
            log_warning("Failed to generate lid_lift_qpos")
            return False
        env.affordance_datas["lid_lift_qpos"] = lid_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_aside_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        lid_aside_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_lift_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_aside_pose is None:
            return False
        env.affordance_datas["lid_aside_pose"] = lid_aside_pose

        lid_aside_qpos = validation_with_process_from_name(
            env,
            lid_aside_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_aside_qpos is None:
            log_warning("Failed to generate lid_aside_qpos")
            return False
        env.affordance_datas["lid_aside_qpos"] = lid_aside_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_lid_back_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        # Generate lid back pose (place lid back on pan) - using v1 logic
        # Start from lid lift pose and lower it by 0.09m like in v1
        lid_back_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["lid_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if lid_back_pose is None:
            return False
        env.affordance_datas["lid_back_pose"] = lid_back_pose

        lid_back_qpos = validation_with_process_from_name(
            env,
            lid_back_pose,
            valid_funcs_name_kwargs_proc[1:],
        )
        if lid_back_qpos is None:
            log_warning("Failed to generate lid_back_qpos")
            return False
        env.affordance_datas["lid_back_qpos"] = lid_back_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def left_arm_compute_unoffset_for_exp(
        env: OpenPanPickAndPlaceEnv, pose_input_output_names_changes: Dict = {}
    ):
        env.affordance_datas["lid_grasp_unoffset_matrix_object"] = np.eye(4)

        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    ########################## right arm ##################################

    @staticmethod
    @tag_node
    def generate_right_arm_init_qpos(env: OpenPanPickAndPlaceEnv):
        right_arm_init_pose = env._get_arm_fk(
            env.affordance_datas["right_arm_init_qpos"],
            uid="right_arm",
            is_world_coordinates=True,
        )
        env.affordance_datas["right_arm_init_pose"] = np.array(right_arm_init_pose)
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_grasp_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.apple_grasp_pose_object),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_grasp_pose is None:
            return False
        env.affordance_datas["apple_grasp_pose"] = apple_grasp_pose

        apple_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_grasp_qpos is None:
            return False
        env.affordance_datas["apple_grasp_qpos"] = apple_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: process pose & valid & process qpos
    def generate_apple_pre1_grasp_pose_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_pre1_grasp_pose is None:
            return False
        env.affordance_datas["apple_pre1_grasp_pose"] = apple_pre1_grasp_pose

        apple_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_pre1_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_pre1_grasp_qpos is None:
            return False
        env.affordance_datas["apple_pre1_grasp_qpos"] = apple_pre1_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_pre2_grasp_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_pre2_grasp_pose is None:
            return False
        env.affordance_datas["apple_pre2_grasp_pose"] = apple_pre2_grasp_pose

        apple_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_pre2_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_pre2_grasp_qpos is None:
            log_warning("Failed to generate apple_pre2_grasp_qpos")
            return False
        env.affordance_datas["apple_pre2_grasp_qpos"] = apple_pre2_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_lift_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_grasp_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_lift_pose is None:
            return False
        env.affordance_datas["apple_lift_pose"] = apple_lift_pose

        apple_lift_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_lift_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_lift_qpos is None:
            log_warning("Failed to generate apple_lift_qpos")
            return False
        env.affordance_datas["apple_lift_qpos"] = apple_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_to_pan_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_to_pan_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_lift_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_to_pan_pose is None:
            return False
        env.affordance_datas["apple_to_pan_pose"] = apple_to_pan_pose

        apple_to_pan_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_to_pan_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_to_pan_qpos is None:
            log_warning("Failed to generate apple_to_pan_qpos")
            return False
        env.affordance_datas["apple_to_pan_qpos"] = apple_to_pan_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_apple_after_pan_qpos(
        env: OpenPanPickAndPlaceEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        apple_after_pan_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_to_pan_pose"]),  # NOTE
            valid_funcs_name_kwargs_proc[:1],
        )
        if apple_after_pan_pose is None:
            return False
        env.affordance_datas["apple_after_pan_pose"] = apple_after_pan_pose

        apple_after_pan_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["apple_after_pan_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if apple_after_pan_qpos is None:
            log_warning("Failed to generate apple_after_pan_qpos")
            return False
        env.affordance_datas["apple_after_pan_qpos"] = apple_after_pan_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def right_arm_compute_unoffset_for_exp(
        env: OpenPanPickAndPlaceEnv, pose_input_output_names_changes: Dict = {}
    ):
        env.affordance_datas["apple_grasp_unoffset_matrix_object"] = np.eye(4)

        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    @staticmethod
    @tag_edge
    @tag_node
    # TODO: Got the dimension from the scope
    def execute_open(
        env: OpenPanPickAndPlaceEnv, return_action: bool = False, **kwargs
    ):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[1.0]])]).transpose()
            else:
                action = np.ones((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(
        env: OpenPanPickAndPlaceEnv, return_action: bool = False, **kwargs
    ):

        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
                action = np.concatenate([action, np.array([[0.0]])]).transpose()
            else:
                action = np.zeros((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        stand_still_qpos = keyposes[0]

        if stand_still_qpos.shape != env.agent.get_joint_ids("left_arm").shape:
            log_error(
                f"The shape of stand_still qpos is different from {agent_uid}'s setting."
            )

        if any(
            np.linalg.norm(former - latter).sum() > 1e-6
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            log_warning(
                f"Applying stand still to two different qpos! Using the first qpos {stand_still_qpos}"
            )
            keyposes = [stand_still_qpos] * 2

        ret = np.asarray([stand_still_qpos] * duration)

        return ret.T

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        if all(
            np.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            log_warning(
                f"Applying plan_trajectory to two very close qpos! Using stand_still."
            )
            keyposes = [keyposes[0]] * 2
            ret_transpoesd = OpenPanActionBank.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
            )

            return ret_transpoesd

        else:
            ret, _ = ArmAction.create_discrete_trajectory(
                agent=env.agent,
                uid=agent_uid,
                qpos_list=keyposes,
                sample_num=duration,
                qpos_seed=keyposes[0],
                is_use_current_qpos=False,
                **env.planning_config,
            )

            return ret.T
