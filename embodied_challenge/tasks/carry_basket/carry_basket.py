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


@register_env("CarryBasket-v1", max_episode_steps=600)
class CarryBasketEnv(ManipulationEnv):
    def __init__(
        self,
        num_envs=1,
        render_backend=False,
        headless=False,
        enable_rt: bool = False,
        **kwargs,
    ):
        from rlia.kit.utility import TrajectorySampleMethod
        from rlia.kit.drive_controllers.utility import PathPlanningType

        self.affordance_datas = {}
        self.planning_config = {
            "sample_method": TrajectorySampleMethod.QUANTITY,
            "is_linear": False,
        }
        self.basket_grasp_pose_object = np.eye(4)
        self.basket_place_pose = np.eye(4)
        self.milk_grasp_pose_object = np.eye(4)
        self.milk_place_pose = np.eye(4)
        self.b_fail = 0
        self.c_fail = 0

        self.camera_dict = {}
        self.dataset_uuid_dict = {}
        self.flag = False
        self.total_nums = 0
        self.supported_robots += ["CobotMagic", "DexAloha"]

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
        self._default_physical_attr.static_friction = 0.95
        self._default_physical_attr.dynamic_friction = 0.9
        self._default_physical_attr.linear_damping = 0.9
        self._default_physical_attr.angular_damping = 0.9
        self._default_physical_attr.contact_offset = 0.005
        self._default_physical_attr.rest_offset = 0.001
        self._default_physical_attr.restitution = 0.0
        self._default_physical_attr.max_depenetration_velocity = 1e0
        self._default_physical_attr.max_linear_velocity = 1e0
        self._default_physical_attr.max_angular_velocity = 1e0

        self.table = self.scene.get_fixed_actor(
            self.scene.get_fixed_actor_uid_list()[0]
        )
        table_attr = self._default_physical_attr
        table_attr.mass = 1.0
        self.table.set_physical_attr(table_attr)
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

        # TODO: 重新安排位置
        action_config = kwargs.get("action_config", None)
        if action_config is None:
            log_error(
                f"The action config is None, but it's needed for Env: {type(self).__name__}, Task Type: {self.metadata['task_type']}."
            )
        self.action_bank = CarryBasketActionBank(action_config)

        vis_graph = action_config.get("other", {}).get("vis_graph", False)
        self.graph_compose, jobs_data, jobkey2index = self.action_bank.parse_network(
            get_func_tag("node").functions[self.action_bank.__class__.__name__],
            get_func_tag("edge").functions[self.action_bank.__class__.__name__],
            vis_graph=vis_graph,
        )

        vis_gantt = action_config.get("other", {}).get("vis_gantt", False)
        self.packages = self.action_bank.gantt(jobs_data, jobkey2index, vis=vis_gantt)

    def _setup_scene(self, **kwargs):
        # TODO: find a more elegent way of doing so
        self.analytic_generate_traj = self.metadata["robot_action"].get(
            "analytic_generate_traj", True
        )

        self.is_dual_arm = self.metadata["robot_action"].get("dual_arm", False)

        super()._setup_scene(**kwargs)

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
        self.basket_object_candidates = [
            obj for obj in self.metadata["objects"] if obj.name == "basket"
        ]
        self.milk_object_candidates = [
            obj for obj in self.metadata["objects"] if obj.name == "milk"
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

            basket_mat = np.random.choice(
                self.plastic_mat_list, size=1, replace=False
            ).tolist()[0]
            milk_mat = np.random.choice(
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
            random_texture_or_uniform_color(basket_mat)
            random_texture_or_uniform_color(milk_mat)

            self.table.set_material(table_mat)
            self.basket.set_material(basket_mat)
            self.milk.set_material(milk_mat)

            articulation = self.get_agent().get_articulation()

            link_names = articulation.get_link_names()
            for link_name in link_names:
                random_texture_or_uniform_color(self.agent_mat, link_name)

    def get_simulated_trajectory_num(self):
        return 1

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
                approach_direction = self.basket_approach_direction
            pre1_pose_world = deepcopy(grasp_world)
            pre1_pose_world[0:3, 3] -= 0.05 * approach_direction
            pre2_pose_world = deepcopy(grasp_world)
            pre2_pose_world[0:3, 3] -= 0.1 * approach_direction

        return grasp_world, pre1_pose_world, pre2_pose_world

    def _setup_simulated_objects_analytic(self, **kwargs):
        # set up basket
        basket = random.choice(self.basket_object_candidates)
        self.basket_xy_random_center = np.asarray(basket.pose)[:2, 3] + [
            sum(self.metadata["robot_action"]["max_random_shift"]["basket"][:2]) / 2,
            sum(self.metadata["robot_action"]["max_random_shift"]["basket"][2:]) / 2,
        ]
        basket_xpos = add_xy_random_offset(
            np.array(basket.pose),
            self.metadata["robot_action"]["max_random_shift"]["basket"],
        )
        standard_basket_verts = apply_svd_transfer_pc(basket)
        self.basket_radius = (
            standard_basket_verts[:, 1].max() - standard_basket_verts[:, 1].min()
        ) / 2

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()
        basket_xpos = basket_xpos @ rotation_matrix

        self.basket_grasp_pose_object = basket.grasp_pose_object
        self.basket_grasp_offset = basket.grasp_offset

        self.basket_grasp_control_part = "right_arm"
        self.basket = self.scene.add_dynamic_actor(
            basket.get_mesh_file(),
            basket_xpos,
            basket.scale,
            compute_uv=True,
            is_convex_decomposition=True,
            max_convex_hull_num=12,
        )
        self.basket_physical_attr = self._default_physical_attr
        self.basket_physical_attr.mass = 0.5
        self.basket.set_physical_attr(self.basket_physical_attr)

        # setup milk
        milk = random.choice(self.milk_object_candidates)
        self.milk_xy_random_center = np.asarray(milk.pose)[:2, 3] + [
            sum(self.metadata["robot_action"]["max_random_shift"]["milk"][:2]) / 2,
            sum(self.metadata["robot_action"]["max_random_shift"]["milk"][2:]) / 2,
        ]
        milk_xpos = add_xy_random_offset(
            np.array(milk.pose),
            self.metadata["robot_action"]["max_random_shift"]["milk"],
        )

        rotation_range = self.metadata["robot_action"].get(
            "rotation_range",
            [[0, 0], [0, 0], [-22.5, 22.5]]
            # NOTE: This is world axes. as rot @ pose
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
        milk_xpos[:3, :3] = rot.as_matrix() @ milk_xpos[:3, :3]

        standard_milk_verts = apply_svd_transfer_pc(milk)
        self.milk_half_height = (
            standard_milk_verts[:, 0].max() - standard_milk_verts[:, 0].min()
        ) / 2

        self.milk_grasp_pose_object = milk.grasp_pose_object
        self.milk_grasp_offset = milk.grasp_offset

        self.milk = self.scene.add_dynamic_actor(
            milk.get_mesh_file(),
            milk_xpos,
            milk.scale,
            compute_uv=True,
            is_convex_decomposition=True,
            max_convex_hull_num=1,
        )
        self.milk_pick_pose_generator = milk.pickpose_sampler
        self.milk_physical_attr = self._default_physical_attr
        self.milk_physical_attr.mass = 0.005
        self.milk.set_physical_attr(self.milk_physical_attr)

        if self.is_do_domain_randomization() and self.is_do_material_randomization():
            self.basket.set_material(
                np.random.choice(self.plastic_mat_list, size=1, replace=False).tolist()[
                    0
                ]
            )
            self.milk.set_material(
                np.random.choice(self.plastic_mat_list, size=1, replace=False).tolist()[
                    0
                ]
            )

        self.scene.update(step=100)

        self.milk_pose_orig = self.milk.get_world_pose()
        self.basket_pose_orig = self.basket.get_world_pose()

    def validate_pose(
        self,
        pose,
        qpos_seed,
        is_left,
        pose_name="pose",
    ):
        ret, qpos = self._get_arm_ik(
            pose,
            is_left=is_left,
            qpos_seed=qpos_seed,
        )
        if ret is False:
            log_warning(f"Generate {pose_name} failed.\n")
            self.b_fail = True
            return None
        if is_qpos_flip(qpos, qpos_seed, self.agent_qpos_flip_ids):
            log_warning(f"{pose_name} flip.\n")
            return None
        return qpos

    def find_nearest_valid_pose(self, pose, select_arm, xpos_resolution=0.02):
        # use the validator to choose the nearest valid pose
        # delete the cache every time
        import shutil
        import os

        home_path = os.path.expanduser("~")
        folder_path = os.path.join(home_path, "embodysim_cache/robot_reachable_xpos")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        ret, _ = self.agent.compute_xpos_reachability(
            select_arm,
            pose,
            xpos_resolution=xpos_resolution,
            qpos_resolution=np.radians(60),
            cache_mode="disk",
            use_cached=False,
        )
        ret = np.stack(ret, axis=0)
        ret = self.agent.get_base_xpos(select_arm) @ ret  # convert to world coordinates
        # find the nearest valid pose
        xyz = pose[:3, 3]
        ts = np.stack([M[:3, 3] for M in ret], axis=0)  # shape (N,3)
        dists = np.linalg.norm(ts - xyz[None, :], axis=1)
        best_idx = np.argmin(dists)
        nearest_valid_pose = ret[best_idx]
        return nearest_valid_pose

    def _setup_simulated_objects(self, **kwargs):
        if self.analytic_generate_traj:
            self._setup_simulated_objects_analytic(**kwargs)
        else:
            log_error(
                "The current carry basket env does not support imitation simulated objs."
            )

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
        if "exteroception" in self.metadata["dataset"]["robot_meta"]["observation"]:
            obs = self._extend_exteroception(obs, arena_index, **kwargs)

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
            "left_arm_milk_grasp_unoffset_matrix_object", np.eye(4)
        )
        right_arm_exp_offset_matrix = self.affordance_datas.get(
            "right_arm_basket_grasp_unoffset_matrix_object", np.eye(4)
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

            if hasattr(self, "basket") and hasattr(self, "milk"):
                self.dataset_uuid_dict["object"] = [
                    self.basket.get_user_id(),
                    self.milk.get_user_id(),
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

    def compute_basket_milk_len_angle(self, basket_pose, milk_pose):
        basket_center = basket_pose[:3, 3]
        milk_center = milk_pose[:3, 3]

        vec = milk_center - basket_center
        length = np.linalg.norm(vec)
        vec_norm = vec / length

        basket_z_axis = basket_pose[:3, 2]
        basket_z_axis_norm = basket_z_axis / np.linalg.norm(basket_z_axis)

        dot_product = np.dot(basket_z_axis_norm, vec_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        return length, angle

    def is_fall(self, pose: np.ndarray):
        # checks whether the object has fallen over by measuring
        # if its local z-axis tilts more than 45° from the world’s vertical axis.
        pose_rz = pose[:3, 2]
        world_z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.dot(pose_rz, world_z_axis))
        if angle >= np.pi / 4:
            return True
        else:
            return False

    def is_task_success(self, obs_list: List) -> bool:
        # sheng: TODO: @runyi needs to check
        """Judge grasp basket and pour water process success.

        Args:
            obs_list (List): The observation list collected from the environment conducting the action list.

        Returns:
            bool: If the task is success given the obs_list
        """

        basket_final_xpos = self.basket.get_world_pose()
        milk_final_xpos = self.milk.get_world_pose()
        basket_xy = basket_final_xpos[:2, 3]
        milk_xy = milk_final_xpos[:2, 3]

        # Check if the basket is placed correctly
        place_basket_xy_loc = self.affordance_datas["right_arm_basket_place_pose"][
            :2, 3
        ]
        tolerance = 0.03
        dist = np.linalg.norm(basket_xy - place_basket_xy_loc)
        if dist > tolerance:
            log_warning(
                f"The basket is not close to the place position in the end, with dist {dist} > tolerance {tolerance}."
            )
            return False

        # Check if milk is inside basket
        basket_radius = self.basket_radius
        dist = np.linalg.norm(milk_xy - basket_xy)
        if dist > basket_radius:
            log_warning(
                f"The milkbox is outside of the basket in the end, with dist {dist} > basket_radius {basket_radius}."
            )
            return False

        # Check if basket is fall
        rot_x = np.eye(4)
        rot_x[:3, :3] = R.from_euler("x", -90, degrees=True).as_matrix()
        basket_final_xpos_align_world = basket_final_xpos @ rot_x
        if self.is_fall(basket_final_xpos_align_world):
            log_warning("Basket has fallen in the end!")
            return False

        return True

    def to_dataset(
        self,
        obs_list: list,
        action_list: list,
        id: str,
        to_camera_frame=True,
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
        from embodychain.utils.visualizer import HeatMapEnv
        import time

        if not self.flag:
            self.map1 = HeatMapEnv(is_success=True)
            self.map2 = HeatMapEnv(is_success=False)
            self.flag = True
        if self.total_nums % 20 == 0:
            self.map1.save_map()
            self.map2.save_map()
        if isinstance(action_list, List):
            action_list[:] = []
            point_b = self.basket.get_world_pose()[0:2, 3]
            self.map1.update_heatmap(point_b, new_fail=2)
            point_c = self.milk.get_world_pose()[0:2, 3]
            self.map1.update_heatmap(point_c, new_fail=2)
            self.total_nums += 1
            time.sleep(0.1)
        else:
            if self.b_fail:
                point_b = self.basket.get_world_pose()[0:2, 3]
                self.map2.update_heatmap(point_b, new_fail=0)
                self.b_fail = False
            if self.c_fail:
                point_c = self.milk.get_world_pose()[0:2, 3]
                self.map2.update_heatmap(point_c, new_fail=1)
                self.c_fail = False
            self.total_nums += 1
            time.sleep(0.1)

    def is_grasp_pose_exceed(
        self,
        grasp_pose: np.ndarray,
        max_z: float,
        half_height: float,
        lower_bound: float = 0.0,
        upper_bound=1.0,
        return_inverse: bool = False,
    ):
        grasp_to_max_z = max_z - grasp_pose[2, 3]
        ret = (grasp_to_max_z <= lower_bound * half_height) or (
            grasp_to_max_z >= upper_bound * half_height
        )

        if return_inverse:
            ret = not ret
        return ret

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
        log_info(f"Milk_pos: {self.milk.get_world_pose()[0:3,3]}.", color="green")
        log_info(f"Basket_pos: {self.basket.get_world_pose()[0:3,3]}.", color="green")
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

    def _get_arm_fk(self, qpos, uid, is_world_coordinates=True):
        return self.agent.get_fk(qpos, uid, is_world_coordinates=is_world_coordinates)


class CarryBasketActionBank(ActionBank):
    """----------------------------------------------Left Arm----------------------------------------------"""

    @staticmethod
    @tag_node
    def generate_left_arm_init_qpos(env: CarryBasketEnv):
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
    def generate_left_arm_milk_grasp_qpos(
        env: CarryBasketEnv, valid_funcs_name_kwargs_proc: List = []
    ):
        left_arm_milk_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.milk_grasp_pose_object),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_grasp_pose is None:
            return False
        env.affordance_datas["left_arm_milk_grasp_pose"] = left_arm_milk_grasp_pose

        left_arm_milk_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_grasp_qpos is None:
            return False
        env.affordance_datas["left_arm_milk_grasp_qpos"] = left_arm_milk_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: qpos ik & valid
    def generate_left_arm_milk_pre2_grasp_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        left_arm_milk_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_pre2_grasp_pose is None:
            return False
        env.affordance_datas[
            "left_arm_milk_pre2_grasp_pose"
        ] = left_arm_milk_pre2_grasp_pose

        left_arm_milk_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_pre2_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_pre2_grasp_qpos is None:
            return False
        env.affordance_datas[
            "left_arm_milk_pre2_grasp_qpos"
        ] = left_arm_milk_pre2_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: qpos ik & valid
    def generate_left_arm_milk_pre1_grasp_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        left_arm_milk_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_pre1_grasp_pose is None:
            return False
        env.affordance_datas[
            "left_arm_milk_pre1_grasp_pose"
        ] = left_arm_milk_pre1_grasp_pose

        left_arm_milk_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_pre1_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_pre1_grasp_qpos is None:
            return False
        env.affordance_datas[
            "left_arm_milk_pre1_grasp_qpos"
        ] = left_arm_milk_pre1_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_milk_lift_qpos(
        env: CarryBasketEnv, valid_funcs_name_kwargs_proc: List = []
    ):
        left_arm_milk_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_lift_pose is None:
            return False
        env.affordance_datas["left_arm_milk_lift_pose"] = left_arm_milk_lift_pose

        left_arm_milk_lift_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_lift_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_lift_qpos is None:
            return False
        env.affordance_datas["left_arm_milk_lift_qpos"] = left_arm_milk_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_milk_move_basket_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        left_arm_milk_move_basket_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_lift_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_move_basket_pose is None:
            return False
        env.affordance_datas[
            "left_arm_milk_move_basket_pose"
        ] = left_arm_milk_move_basket_pose

        left_arm_milk_move_basket_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_move_basket_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_move_basket_qpos is None:
            return False
        env.affordance_datas[
            "left_arm_milk_move_basket_qpos"
        ] = left_arm_milk_move_basket_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_milk_place_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        left_arm_milk_place_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_move_basket_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if left_arm_milk_place_pose is None:
            return False
        env.affordance_datas["left_arm_milk_place_pose"] = left_arm_milk_place_pose

        left_arm_milk_place_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_milk_place_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if left_arm_milk_place_qpos is None:
            return False
        env.affordance_datas["left_arm_milk_place_qpos"] = left_arm_milk_place_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_left_arm_final_monitor_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        left_arm_final_monitor_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_init_qpos"]),
            valid_funcs_name_kwargs_proc[:-1],
        )
        if left_arm_final_monitor_qpos is None:
            return False
        env.affordance_datas[
            "left_arm_final_monitor_qpos"
        ] = left_arm_final_monitor_qpos

        left_arm_final_monitor_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["left_arm_final_monitor_qpos"]),
            valid_funcs_name_kwargs_proc[-1:],
        )
        if left_arm_final_monitor_pose is None:
            return False
        env.affordance_datas[
            "left_arm_final_monitor_pose"
        ] = left_arm_final_monitor_pose
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def left_arm_compute_unoffset_for_exp(
        env: CarryBasketEnv, pose_input_output_names_changes: Dict = {}
    ):
        env.affordance_datas["left_arm_milk_grasp_unoffset_matrix_object"] = np.eye(4)

        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    """----------------------------------------------Right Arm----------------------------------------------"""

    @staticmethod
    @tag_node
    def generate_right_arm_init_qpos(env: CarryBasketEnv):
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
    def generate_right_arm_basket_grasp_qpos(
        env: CarryBasketEnv, valid_funcs_name_kwargs_proc: List = []
    ):
        right_arm_basket_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.basket_grasp_pose_object),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_grasp_pose is None:
            return False
        env.affordance_datas[
            "right_arm_basket_grasp_pose"
        ] = right_arm_basket_grasp_pose

        right_arm_basket_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_grasp_qpos is None:
            return False
        env.affordance_datas[
            "right_arm_basket_grasp_qpos"
        ] = right_arm_basket_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: qpos ik & valid
    def generate_right_arm_basket_pre2_grasp_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        right_arm_basket_pre2_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_pre2_grasp_pose is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre2_grasp_pose"
        ] = right_arm_basket_pre2_grasp_pose

        right_arm_basket_pre2_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_pre2_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_pre2_grasp_qpos is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre2_grasp_qpos"
        ] = right_arm_basket_pre2_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: qpos ik & valid
    def generate_right_arm_basket_pre1_grasp_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        right_arm_basket_pre1_grasp_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_pre1_grasp_pose is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre1_grasp_pose"
        ] = right_arm_basket_pre1_grasp_pose

        right_arm_basket_pre1_grasp_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_pre1_grasp_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_pre1_grasp_qpos is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre1_grasp_qpos"
        ] = right_arm_basket_pre1_grasp_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_basket_lift_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        right_arm_basket_lift_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_grasp_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_lift_pose is None:
            return False
        env.affordance_datas["right_arm_basket_lift_pose"] = right_arm_basket_lift_pose

        right_arm_basket_lift_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_lift_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_lift_qpos is None:
            return False
        env.affordance_datas["right_arm_basket_lift_qpos"] = right_arm_basket_lift_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_basket_place_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        right_arm_basket_place_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_lift_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_place_pose is None:
            return False
        env.affordance_datas[
            "right_arm_basket_place_pose"
        ] = right_arm_basket_place_pose

        right_arm_basket_place_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_place_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_place_qpos is None:
            return False
        env.affordance_datas[
            "right_arm_basket_place_qpos"
        ] = right_arm_basket_place_qpos

        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_right_arm_basket_pre_place_qpos(
        env: CarryBasketEnv,
        valid_funcs_name_kwargs_proc: List = [],
    ):
        right_arm_basket_pre_place_pose = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_place_pose"]),
            valid_funcs_name_kwargs_proc[:1],
        )
        if right_arm_basket_pre_place_pose is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre_place_pose"
        ] = right_arm_basket_pre_place_pose

        right_arm_basket_pre_place_qpos = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas["right_arm_basket_pre_place_pose"]),
            valid_funcs_name_kwargs_proc[1:],
        )
        if right_arm_basket_pre_place_qpos is None:
            return False
        env.affordance_datas[
            "right_arm_basket_pre_place_qpos"
        ] = right_arm_basket_pre_place_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    def right_arm_compute_unoffset_for_exp(
        env: CarryBasketEnv, pose_input_output_names_changes: Dict = {}
    ):
        env.affordance_datas["right_arm_basket_grasp_unoffset_matrix_object"] = np.eye(
            4
        )

        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    """-------------------------------------------Useful Functions-------------------------------------------"""

    @staticmethod
    @tag_edge
    @tag_node
    # TODO: Got the dimension from the scope
    def execute_open(env: CarryBasketEnv, return_action: bool = False, **kwargs):
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
    def execute_close(env: CarryBasketEnv, return_action: bool = False, **kwargs):

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
            ret_transpoesd = CarryBasketActionBank.stand_still(
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


def visual_action(self, action_list: list, visual_time: float = 3.0) -> None:
    if action_list is not None:
        action_list[:] = []
        # TODO
