"""Ground the challenge drawer's handle marker for EmbodiChain's Slide skill."""

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

    def __init__(self, semantics):
        self.semantics = semantics

    def lower(self, call, *, context, bound, option_template):
        if call.arguments["handle"] != self.semantics.entity_id:
            raise ValueError("The drawer Slide call must name drawer_tray.")
        return SemanticLowering(
            goal=SlideGoal(
                self.semantics,
                context.scene.entities["drawer_tray"].pose.clone(),
                joint_target=SlideJointTarget(
                    "drawer", "slide_rails", call.arguments["position"],
                    axis_sign=self.axis_sign,
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
        handle[:3, :3] = torch.as_tensor(
            Rotation.from_euler("YZX", (90, -90, -35), degrees=True).as_matrix(),
            dtype=tray.dtype,
            device=tray.device,
        )
        vertices, faces = drawer.get_link_vert_face("inner_box")
        affordance = DrawerHandleAffordance(
            custom_config={"handle_grasp": torch.linalg.solve(tray, handle)},
            mesh_vertices=torch.as_tensor(
                vertices, dtype=torch.float32, device=tray.device
            ),
            mesh_triangles=torch.as_tensor(faces, dtype=torch.long, device=tray.device),
            translation_axis=tray.new_tensor(self.rail_axis),
            joint_name="slide_rails",
            joint_limits=(0.0, 0.20),
        )
        return self.lowerer_type(
            ObjectSemantics(affordance=affordance, geometry={}, entity_id="drawer_tray")
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
