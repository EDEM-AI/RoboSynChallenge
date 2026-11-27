"""Ground the challenge drawer's handle marker for EmbodiChain's Slide skill."""

from dataclasses import replace

import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_skills import (
    ObjectSemantics,
    Place,
    PlaceGoal,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideJointTarget,
)
from embodichain.lab.sim.atomic_skills.affordance_sampling import (
    AffordancePoseCandidates,
)
from embodichain.lab.task_program.compiler.lowering import (
    HeldObjectRelation,
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticEffectKind,
    SemanticLowering,
)
from embodichain.lab.task_program.language.loader import load_task_program
from .task_program_adapter import compose_task_program


class DrawerHandleAffordance(SlideAffordance):
    """Use the authored handle frame as the calibrated grasp candidate."""

    def get_grasp_candidates(
        self, generator, object_poses, approach_direction, **kwargs
    ):
        handle_grasp = self.custom_config["handle_grasp"].to(object_poses)
        poses = (object_poses @ handle_grasp).unsqueeze(1)
        return AffordancePoseCandidates(
            poses=poses,
            costs=poses.new_zeros(poses.shape[:2]),
            valid=torch.isfinite(poses).all(dim=(-1, -2)),
        )


class DrawerSlideLowerer(RegisteredSemanticLowerer):
    call_id = "simulation.articulation_link_slide"
    target_descriptor = Slide.descriptor()
    requires_unoccupied_slots = ("primary",)
    preserves_symbolic_state = True
    axis_sign = -1

    def __init__(self, semantics, robot):
        self.semantics = semantics
        self.robot = robot

    def _select_handle_grasp(self, context, bound, options, tray, displacement):
        affordance = self.semantics.affordance
        arm = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        grasps = tray[:, None] @ affordance.custom_config["handle_grasps"].to(tray)
        rail = tray[:, :3, :3] @ affordance.translation_axis.to(tray)
        approach = (
            grasps[..., :3, 2] if options.approach_along_grasp_axis else rail[:, None]
        )
        start = grasps[..., :3, 3]
        stroke = rail[:, None] * displacement[:, None, None]
        end = start + stroke
        reach = approach * options.approach_distance
        retreat = grasps[..., :3, 2] * options.release_retreat_distance
        fractions = torch.linspace(0, 1, 13, device=tray.device, dtype=tray.dtype)
        waypoints = torch.cat(
            (
                start[:, :, None] - reach[:, :, None] * (1 - fractions[:, None]),
                start[:, :, None] + stroke[:, :, None] * fractions[1:, None],
                end[:, :, None] - retreat[:, :, None] * fractions[3::3, None],
            ),
            dim=2,
        )
        poses = grasps[:, :, None].expand(-1, -1, waypoints.shape[2], -1, -1).clone()
        poses[..., :3, 3] = waypoints
        count = grasps.shape[1]
        joints = context.robot.qpos[:, list(arm.joint_ids)]
        env_ids = torch.as_tensor(context.env_ids, device=tray.device)
        # Give every candidate its own seed so the native solver tracks one
        # continuous IK branch along each reach, rail stroke and retreat.
        valid, solutions = self.robot.compute_batch_ik(
            pose=poses.flatten(0, 1),
            joint_seed=joints.repeat_interleave(count, dim=0),
            name=arm.control_part,
            env_ids=env_ids.repeat_interleave(count),
            continuous=True,
        )
        valid = valid.reshape(context.batch_size, count, -1).all(dim=-1)
        solutions = solutions.reshape(context.batch_size, count, waypoints.shape[2], -1)
        step_cost = solutions.diff(dim=2).abs().amax(dim=(-1, -2))
        start_cost = (solutions[:, :, 0] - joints[:, None]).abs().amax(dim=-1)
        cost = (step_cost + 0.1 * start_cost).masked_fill(~valid, torch.inf)
        rows = torch.arange(context.batch_size, device=tray.device)
        selected = grasps[rows, cost.argmin(dim=1)]
        return torch.linalg.solve(tray, selected)

    def lower(self, call, *, context, bound, option_template):
        if call.arguments["handle"] != self.semantics.entity_id:
            raise ValueError("The drawer Slide call must name drawer_tray.")
        tray = context.scene.entities["drawer_tray"].pose
        joint = context.scene.articulation_joints[("drawer", "slide_rails")]
        position = joint.position.reshape(-1).expand(context.batch_size)
        target = call.arguments["position"]
        grasp = self._select_handle_grasp(
            context, bound, option_template, tray, (target - position) * self.axis_sign
        )
        affordance = replace(
            self.semantics.affordance, custom_config={"handle_grasp": grasp}
        )
        return SemanticLowering(
            goal=SlideGoal(
                replace(self.semantics, affordance=affordance),
                tray.clone(),
                joint_target=SlideJointTarget(
                    "drawer", "slide_rails", target, axis_sign=self.axis_sign,
                ),
            )
        )


class DrawerSlideFactory:
    call_id = "simulation.articulation_link_slide"
    revision = "1"
    target_descriptor = Slide.descriptor()
    lowerer_type = DrawerSlideLowerer
    handle_height_offset = -0.022
    handle_rail_offset = 0.0
    rail_axis = (-1.0, 0.0, 0.0)

    def create(self, *, simulation, robot, scene_registry, engine):
        drawer = simulation.get_articulation("drawer")
        tray = drawer.get_link_pose("inner_box", to_matrix=True)[0]
        handle = drawer.get_link_pose("handle_xpos", to_matrix=True)[0].clone()
        # The locator is above the graspable rail lip in the scaled scene.
        handle[2, 3] += self.handle_height_offset
        handle[:3, 3] -= self.handle_rail_offset * tray[:3, 0]
        handles = handle.expand(5, -1, -1).clone()
        handles[:, :3, :3] = torch.as_tensor(
            Rotation.from_euler(
                "YZX", [(90, -90, angle) for angle in (-35, -25, -45, -15, -55)],
                degrees=True,
            ).as_matrix(),
            dtype=tray.dtype,
            device=tray.device,
        )
        vertices, faces = drawer.get_link_vert_face("inner_box")
        affordance = DrawerHandleAffordance(
            custom_config={"handle_grasps": torch.linalg.solve(tray, handles)},
            mesh_vertices=torch.as_tensor(
                vertices, dtype=torch.float32, device=tray.device
            ),
            mesh_triangles=torch.as_tensor(faces, dtype=torch.long, device=tray.device),
            translation_axis=tray.new_tensor(self.rail_axis),
            joint_name="slide_rails",
            joint_limits=(0.0, 0.20),
        )
        return self.lowerer_type(
            ObjectSemantics(affordance=affordance, geometry={}, entity_id="drawer_tray"),
            robot,
        )


class DrawerCloseLowerer(DrawerSlideLowerer):
    call_id = "robosyn.close_drawer"
    axis_sign = 1


class DrawerCloseFactory(DrawerSlideFactory):
    """Encode inward travel as a negative stroke with a short release."""

    call_id = DrawerCloseLowerer.call_id
    lowerer_type = DrawerCloseLowerer
    handle_rail_offset = 0.015
    rail_axis = (1.0, 0.0, 0.0)


class DrawerPlace(RegisteredSemanticLowerer):
    call_id = "simulation.place_relative"
    target_descriptor = Place.descriptor()
    effect_contract_kind = SemanticEffectKind.RELEASE

    def __init__(self, arm_base):
        self.arm_base = arm_base

    def lower(self, call, *, context, bound, option_template):
        hand = bound.binding.resources["primary"]
        held = context.task.get_held_object(hand.endpoints["motion"].task_state_key)
        if held is None or held.semantics.entity_id != "tomato":
            raise ValueError("Drawer placement requires the tomato in the left hand.")
        tray = context.scene.entities["drawer_tray"].pose
        target = (tray @ tray.new_tensor([0.02, -0.02, 0.10, 1.0]))[:, :3]
        direction = target - self.arm_base.to(target)
        heading = torch.atan2(direction[:, 1], direction[:, 0])
        angles = torch.stack(
            (
                heading,
                torch.full_like(heading, -torch.pi / 3),
                torch.full_like(heading, torch.pi),
            ),
            dim=-1,
        )
        rotation = torch.as_tensor(
            Rotation.from_euler("ZYX", angles.cpu().numpy()).as_matrix(),
            dtype=tray.dtype,
            device=tray.device,
        )
        # Use the legacy wrist posture and compensate for the selected grasp.
        tcp_to_tomato = torch.linalg.inv(held.object_to_eef.to(tray))
        release = tray.clone()
        release[:, :3, :3] = rotation
        release[:, :3, 3] = target - (
            rotation @ tcp_to_tomato[..., :3, 3, None]
        ).squeeze(-1)
        return SemanticLowering(
            goal=PlaceGoal(release),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.RELEASE,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.DETACHED,
                        object_id="tomato",
                        slot_id="primary",
                        release_before_segment="retract",
                        retain_during_segments=("approach",),
                    ),
                ),
            ),
        )


class DrawerPlaceFactory:
    call_id = DrawerPlace.call_id
    revision = "1"
    target_descriptor = DrawerPlace.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        return DrawerPlace(robot.get_link_pose("left_arm_base")[:, :3].clone())


_DEPLOYMENT, _ADAPTER = compose_task_program(
    "drawer_open_place",
    (DrawerSlideFactory(), DrawerPlaceFactory()),
    additional_lowerers=(DrawerCloseFactory(),),
)
_PROGRAM = load_task_program(
    _DEPLOYMENT.program_path,
    integration=_DEPLOYMENT.selection,
    validation_context=_ADAPTER.registration.catalog,
)


@register_env(
    "RoboSyn-DrawerOpenPlace-v1",
    max_episode_steps=1400,
    task_program_registration=_ADAPTER.registration,
    task_program_adapter_factory=_ADAPTER,
)
class DrawerOpenPlace(EmbodiedEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.task_program is None:
            cfg.task_program = _PROGRAM
        super().__init__(cfg=cfg, **kwargs)
        self._peak_rail = torch.zeros(self.num_envs, device=self.device)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._peak_rail[(options or {}).get("reset_ids", slice(None))] = 0.0
        return obs, info

    def _drawer_metrics(self):
        drawer = self.sim.get_articulation("drawer")
        tomato = self.sim.get_rigid_object("duck").get_local_pose()
        drawer_base = drawer.get_link_pose("outer_box")
        distance = torch.linalg.vector_norm(tomato[:, :2] - drawer_base[:, :2], dim=-1)
        return distance, drawer.get_qpos()[:, 0]

    def compute_task_state(self, **kwargs):
        distance, rail = self._drawer_metrics()
        self._peak_rail = torch.maximum(self._peak_rail, rail)
        keep_running = torch.zeros_like(distance, dtype=torch.bool)
        return (
            keep_running,
            keep_running,
            {
                "tomato_drawer_distance_m": distance,
                "drawer_rail_position_m": rail,
                "drawer_peak_rail_position_m": self._peak_rail,
            },
        )

    def is_task_success(self, **kwargs):
        distance, rail = self._drawer_metrics()
        return (distance <= 0.10) & (rail <= 0.02) & (self._peak_rail >= 0.09)
