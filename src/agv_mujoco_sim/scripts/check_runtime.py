#!/usr/bin/env python3
"""Fail-fast runtime check for the AGV MuJoCo/Nav2/RL workspace."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from importlib import metadata


MODULES = (
    "numpy",
    "mujoco",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "rclpy",
)

ROS_PACKAGES = (
    "agv_mujoco_sim",
    "nav2_bringup",
    "nav2_controller",
    "nav2_velocity_smoother",
    "robot_localization",
)


def version_of(module_name: str) -> str:
    try:
        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {},
        "ros_packages": {},
        "errors": [],
    }

    errors: list[str] = result["errors"]  # type: ignore[assignment]
    modules: dict[str, str] = result["modules"]  # type: ignore[assignment]
    ros_packages: dict[str, str] = result["ros_packages"]  # type: ignore[assignment]

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            modules[module_name] = version_of(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            modules[module_name] = "MISSING"
            errors.append(f"Python module {module_name}: {exc}")

    try:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["torch_cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        result["torch_cuda_available"] = False

    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_prefix,
        )

        for package_name in ROS_PACKAGES:
            try:
                ros_packages[package_name] = get_package_prefix(package_name)
            except PackageNotFoundError as exc:
                ros_packages[package_name] = "MISSING"
                errors.append(f"ROS package {package_name}: {exc}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        errors.append(f"ament_index_python unavailable: {exc}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        print("RUNTIME CHECK: FAIL", file=sys.stderr)
        return 1

    print("RUNTIME CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
