"""Grasp geometry for the CobotMagic source asset shipped by RoboSynChallenge."""

ROBOT_CATALOG = {
    "RoboSynCobotMagicV100Gripper": {
        "class_path": "embodichain_robots.agilex.cobotmagic.gripper.cfg:CobotMagicV100GripperCfg",
        "description_path": "robosynchallenge.robot_catalog:localized_gripper_defaults",
    }
}


def localized_gripper_defaults(variant=None):
    from embodichain.robot_registry import resolve_gripper_defaults

    description = resolve_gripper_defaults("CobotMagicV100Gripper", variant=variant)
    description["catalog_id"] = "RoboSynCobotMagicV100Gripper"
    description["source_sha256"] = "7bf80727a9f82707373bb0d3b515ceb020b82b6cffb10ba0d47a91539bddad51"
    return description
