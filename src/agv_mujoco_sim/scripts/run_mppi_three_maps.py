#!/usr/bin/env python3
"""Run MPPI, shielded SAC, or SAC-only speed-cap encounters on three maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


MAPS = (
    ("mppi_warehouse_aisles", "mppi_warehouse_aisles.xml", "mppi_warehouse_aisles.yaml", "warehouse"),
    ("mppi_warehouse_mixed", "mppi_warehouse_mixed.xml", "mppi_warehouse_mixed.yaml", "warehouse"),
    ("mppi_floor2_corridor", "mppi_floor2_corridor.xml", "mppi_floor2_corridor.yaml", "corridor"),
)
BAG_TOPICS = (
    "/clock", "/tf", "/tf_static", "/scan", "/odometry/filtered",
    "/wheel/odom", "/ground_truth/odom", "/cmd_vel_nav", "/cmd_vel",
    "/collision_state", "/collision_event", "/dynamic_obstacle/position",
    "/amcl_pose", "/plan", "/speed_limit", "/rl_velocity_limits",
    "/rl_velocity_limits_applied", "/sim_front_safety_stop",
    "/sim_front_safety_diagnostics",
)

TF_STALE_LINE = re.compile(
    r"\[(\d+(?:\.\d+)?)\] \[tf_help\]: Data time: "
    r"(\d+)s (\d+)ns, Transform time: (\d+)s (\d+)ns"
)

SCENARIO_KEYS = (
    "trigger_x", "obstacle_x", "obstacle_start_y",
    "obstacle_end_y", "obstacle_speed_m_s",
)


def load_replay_scenarios(report_path: Path, trial_ids: list[int],
                          map_id: str) -> list[dict[str, float]]:
    """Load exact scenarios from a prior map report, preserving trial IDs."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata_path = report_path.parent / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("map_id") != map_id:
            raise ValueError(f"Replay report map is not {map_id}")
    elif report_path.parent.name != map_id:
        raise ValueError(f"Replay report must be inside the {map_id} directory")
    by_id = {trial["trial"]: trial for trial in report["trials"]}
    if len(by_id) != len(report["trials"]):
        raise ValueError("Replay report contains duplicate trial IDs")
    scenarios = []
    for trial_id in trial_ids:
        if trial_id not in by_id:
            raise ValueError(f"Trial {trial_id} is absent from {report_path}")
        source = by_id[trial_id]["scenario"]
        scenario = {key: float(source[key]) for key in SCENARIO_KEYS}
        if not all(math.isfinite(value) for value in scenario.values()):
            raise ValueError(f"Trial {trial_id} has a non-finite scenario value")
        if scenario["obstacle_speed_m_s"] <= 0:
            raise ValueError(f"Trial {trial_id} has invalid obstacle speed")
        scenarios.append(scenario)
    return scenarios


class TfStaleMonitor:
    """Watch the controller's own TF diagnostic, not merely the /tf topic."""

    def __init__(self, log_path: Path, start_offset: int = 0) -> None:
        self.log_path = log_path
        self.offset = start_offset
        self.first_error_time = None
        self.last_error_time = None
        self.count = 0

    def poll(self) -> dict | None:
        with self.log_path.open("r", errors="replace") as stream:
            stream.seek(self.offset)
            lines = stream.readlines()
            self.offset = stream.tell()
        for line in lines:
            match = TF_STALE_LINE.search(line)
            if match is None:
                continue
            wall_time = float(match[1])
            data_time = int(match[2]) + int(match[3]) * 1e-9
            transform_time = int(match[4]) + int(match[5]) * 1e-9
            lag = data_time - transform_time
            if lag < 2.0:
                self.first_error_time = self.last_error_time = None
                self.count = 0
                continue
            if self.last_error_time is None or wall_time - self.last_error_time > 0.5:
                self.first_error_time = wall_time
                self.count = 0
            self.last_error_time = wall_time
            self.count += 1
            if self.count >= 5 and wall_time - self.first_error_time >= 0.4:
                return {
                    "reason": "controller_tf_stale",
                    "data_time_s": data_time,
                    "transform_time_s": transform_time,
                    "lag_s": lag,
                    "wall_time_s": wall_time,
                }
        return None


def stop_group(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print(f"[{name}] did not stop after SIGINT; sending SIGTERM", flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)


def wait_for_mppi(launch: subprocess.Popen, timeout_s: float = 90.0) -> float:
    deadline = time.monotonic() + timeout_s
    vx_max = None
    while time.monotonic() < deadline:
        if launch.poll() is not None:
            raise RuntimeError(f"Nav2 launch exited with code {launch.returncode}")
        if vx_max is None:
            try:
                result = subprocess.run(
                    ["ros2", "param", "get", "/controller_server", "FollowPath.vx_max"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                try:
                    vx_max = float(result.stdout.strip().split()[-1])
                except (ValueError, IndexError):
                    pass
        if vx_max is not None:
            states = []
            for node in ("/controller_server", "/planner_server", "/bt_navigator"):
                try:
                    result = subprocess.run(
                        ["ros2", "lifecycle", "get", node],
                        capture_output=True, text=True, timeout=5, check=False,
                    )
                except subprocess.TimeoutExpired:
                    result = None
                states.append(
                    result is not None and result.returncode == 0
                    and result.stdout.strip().startswith("active")
                )
            if all(states):
                return vx_max
        time.sleep(2)
    raise TimeoutError("Nav2 did not activate controller, planner and navigator within 90 s")


def require_boolean_parameter(node: str, name: str, expected: bool) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["ros2", "param", "get", node, name],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except subprocess.TimeoutExpired:
            time.sleep(1)
            continue
        if result.returncode == 0:
            actual = result.stdout.strip().split()[-1].lower()
            if actual == str(expected).lower():
                return
            raise RuntimeError(f"Expected {node} {name}={expected}, got {result.stdout.strip()}")
        time.sleep(1)
    raise RuntimeError(f"Could not verify {node} {name}")


def run_tf_guarded_map(
    args: argparse.Namespace,
    map_id: str,
    model_file: str,
    map_file: str,
    scenario_profile: str,
    launch_cmd: list[str],
    model_path: Path | None,
    map_output: Path,
) -> None:
    """Count only healthy trials; restart the stack and replay a TF-faulted case."""
    from benchmark_dynamic_obstacle import aggregate_trials, trial_scenarios

    scenarios = args.replay_scenarios or trial_scenarios(argparse.Namespace(
        scenario_profile=scenario_profile,
        scenario_seed=args.seed,
        trials=args.trials,
    ))
    trial_ids = args.replay_trial_ids or list(range(1, args.trials + 1))
    completed = []
    faults = []
    attempts = []
    last_report = None
    csv_path = map_output / "benchmark.samples.csv"
    progress_path = map_output / "benchmark.progress.json"
    launch = bag = None
    trial_index = 1
    restart_count = 0
    attempt_number = 0
    try:
        while trial_index <= args.trials:
            attempt_number += 1
            attempt_dir = map_output / f"attempt_{attempt_number:03d}"
            attempt_dir.mkdir()
            launch_log_path = attempt_dir / "launch.log"
            try:
                for bringup_attempt in (1, 2):
                    launch_log_path = attempt_dir / (
                        "launch.log" if bringup_attempt == 1
                        else "launch_retry_2.log"
                    )
                    with launch_log_path.open("w") as launch_log:
                        launch = subprocess.Popen(
                            launch_cmd, stdout=launch_log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    try:
                        vx_max = wait_for_mppi(launch)
                        break
                    except (RuntimeError, TimeoutError) as error:
                        stop_group(launch, "launch")
                        launch = None
                        if bringup_attempt == 2:
                            raise RuntimeError(
                                f"Nav2 bringup failed twice; see {launch_log_path}"
                            ) from error
                        print(f"[{map_id}] bringup failed: {error}; retrying", flush=True)
                        time.sleep(3)
                if abs(vx_max - 0.70) > 1e-3:
                    raise RuntimeError(f"Expected MPPI vx_max=0.70 m/s, got {vx_max}")
                if args.arm == "sac_only":
                    require_boolean_parameter(
                        "/rl_velocity_supervisor", "safety_override_enabled", False,
                    )
                    require_boolean_parameter(
                        "/rl_velocity_supervisor", "ttc_observation_enabled", False,
                    )
                    require_boolean_parameter(
                        "/mujoco_ros_bridge", "front_safety_stop_enabled", False,
                    )
                bag_cmd = [
                    "ros2", "bag", "record", "-o", str(attempt_dir / "rosbag"),
                    *BAG_TOPICS,
                ]
                with (attempt_dir / "rosbag.log").open("w") as bag_log:
                    bag = subprocess.Popen(
                        bag_cmd, stdout=bag_log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                time.sleep(3)
                if bag.poll() is not None:
                    raise RuntimeError(f"rosbag exited with code {bag.returncode}")
                attempts.append({
                    "attempt": attempt_number,
                    "started_at_trial": trial_index,
                    "launch_log": str(launch_log_path),
                    "rosbag": str(attempt_dir / "rosbag"),
                })
                log_offset = 0
                tf_fault = None
                while trial_index <= args.trials:
                    scenario = scenarios[trial_index - 1]
                    trial_id = trial_ids[trial_index - 1]
                    trial_dir = attempt_dir / f"trial_{trial_id:03d}"
                    trial_dir.mkdir()
                    trial_output = trial_dir / "benchmark.json"
                    benchmark_cmd = [
                        "ros2", "run", "agv_mujoco_sim", "benchmark_dynamic_obstacle.py",
                        "--trials", "1", "--timeout-s", str(args.timeout_s),
                        "--output", str(trial_output),
                        "--scenario-profile", "fixed",
                        "--trigger-x", str(scenario["trigger_x"]),
                        "--obstacle-x", str(scenario["obstacle_x"]),
                        "--obstacle-start-y", str(scenario["obstacle_start_y"]),
                        "--obstacle-end-y", str(scenario["obstacle_end_y"]),
                        "--obstacle-speed-m-s", str(scenario["obstacle_speed_m_s"]),
                    ]
                    if args.arm == "shield_only":
                        benchmark_cmd.append("--shield-only")
                    if model_path is not None:
                        benchmark_cmd.extend(("--sac-model", str(model_path)))
                    if args.arm == "sac_only":
                        benchmark_cmd.append("--sac-only")
                    monitor = TfStaleMonitor(launch_log_path, log_offset)
                    with (trial_dir / "benchmark.log").open("w") as benchmark_log:
                        process = subprocess.Popen(
                            benchmark_cmd, stdout=benchmark_log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    while process.poll() is None:
                        tf_fault = monitor.poll()
                        if tf_fault is not None:
                            stop_group(process, "benchmark")
                            break
                        if launch.poll() is not None or bag.poll() is not None:
                            stop_group(process, "benchmark")
                            raise RuntimeError(
                                f"Launch or rosbag died during trial {trial_id}; "
                                f"see {attempt_dir}"
                            )
                        time.sleep(0.2)
                    tf_fault = tf_fault or monitor.poll()
                    log_offset = monitor.offset
                    if tf_fault is not None:
                        tf_fault.update({
                            "trial": trial_id, "attempt": attempt_number,
                            "scenario": scenario,
                            "launch_log": str(launch_log_path),
                            "benchmark_log": str(trial_dir / "benchmark.log"),
                        })
                        faults.append(tf_fault)
                        (map_output / "tf_faults.json").write_text(
                            json.dumps(faults, indent=2) + "\n", encoding="utf-8",
                        )
                        print(
                            f"[{map_id}] stale controller TF in trial {trial_id}; "
                            "discarding this attempt and restarting the stack",
                            flush=True,
                        )
                        break
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"Trial {trial_id} failed ({process.returncode}); "
                            f"see {trial_dir / 'benchmark.log'}"
                        )
                    report = json.loads(trial_output.read_text(encoding="utf-8"))
                    if len(report["trials"]) != 1:
                        raise RuntimeError(f"Expected one trial in {trial_output}")
                    summary = report["trials"][0]
                    if summary["scenario"] != scenario:
                        raise RuntimeError(f"Scenario mismatch in {trial_output}")
                    if summary["elapsed_sim_s"] <= 0:
                        raise RuntimeError(
                            f"Trial {trial_id} has no simulated elapsed time; "
                            f"invalid clock or reset state in {trial_output}"
                        )
                    summary["trial"] = trial_id
                    with trial_output.with_suffix(".samples.csv").open(
                        newline="", encoding="utf-8"
                    ) as source:
                        reader = csv.DictReader(source)
                        fieldnames = reader.fieldnames
                        if not fieldnames or "trial" not in fieldnames:
                            raise RuntimeError(f"Missing trial samples in {trial_output}")
                        with csv_path.open(
                            "w" if trial_index == 1 else "a",
                            newline="", encoding="utf-8",
                        ) as destination:
                            writer = csv.DictWriter(destination, fieldnames=fieldnames)
                            if trial_index == 1:
                                writer.writeheader()
                            for row in reader:
                                row["trial"] = trial_id
                                writer.writerow(row)
                    completed.append(summary)
                    last_report = report
                    progress_path.write_text(json.dumps({
                        "scenario_profile": scenario_profile,
                        "scenario_seed": args.seed,
                        "replay_source": None if args.replay_report is None else str(args.replay_report),
                        "replay_trial_ids": args.replay_trial_ids,
                        "completed_trials": len(completed),
                        "planned_trials": args.trials,
                        "system_tf_faults": faults,
                        "trials": completed,
                        "aggregate_so_far": aggregate_trials(completed),
                    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    print(
                        f"[{map_id}] valid trial {trial_id} ({trial_index}/{args.trials}): "
                        f"{summary['status']}", flush=True,
                    )
                    trial_index += 1
            finally:
                stop_group(bag, "rosbag")
                stop_group(launch, "launch")
                bag = launch = None
                time.sleep(3)
            if trial_index <= args.trials:
                restart_count += 1
                if restart_count > args.max_tf_restarts:
                    raise RuntimeError(
                        f"Controller TF failure repeated; stopped with "
                        f"{len(completed)} valid trials. See {map_output / 'tf_faults.json'}"
                    )
        assert last_report is not None
        last_report["scenario"]["profile"] = scenario_profile
        last_report["scenario"]["seed"] = args.seed
        for varied_key in (
            "trigger_x", "obstacle_start", "obstacle_end_y", "obstacle_speed_m_s"
        ):
            last_report["scenario"].pop(varied_key, None)
        last_report["scenario"]["parameters_in_each_trial"] = True
        if args.replay_report is not None:
            last_report["scenario"]["replay_source"] = str(args.replay_report)
            last_report["scenario"]["replay_trial_ids"] = trial_ids
        last_report["trials"] = completed
        last_report["aggregate"] = aggregate_trials(completed)
        last_report["tf_guarded"] = True
        last_report["system_tf_faults"] = faults
        (map_output / "benchmark.json").write_text(
            json.dumps(last_report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (map_output / "run_metadata.json").write_text(json.dumps({
            "map_id": map_id, "model_file": model_file, "map_file": map_file,
            "arm": args.arm,
            "model": None if model_path is None else str(model_path),
            "trials": args.trials, "scenario_profile": scenario_profile,
            "scenario_seed": args.seed, "vx_max_m_s": vx_max,
            "tf_guarded": True, "system_tf_faults": faults,
            "replay_source": None if args.replay_report is None else str(args.replay_report),
            "replay_trial_ids": args.replay_trial_ids,
            "supervisor_params_file": None if args.supervisor_params_file is None else str(args.supervisor_params_file),
            "attempts": attempts, "aggregate": last_report["aggregate"],
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{map_id}] {json.dumps(last_report['aggregate'])}", flush=True)
    finally:
        stop_group(bag, "rosbag")
        stop_group(launch, "launch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=75.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--arm", choices=("pure_mppi", "shield_only", "sac", "sac_only"),
        default="pure_mppi",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--supervisor-params-file", type=Path, default=None,
        help="optional SAC-only supervisor YAML (baseline used by default)",
    )
    parser.add_argument(
        "--replay-report", type=Path, default=None,
        help="prior map benchmark.json containing exact scenarios to replay",
    )
    parser.add_argument(
        "--replay-trials", type=str, default=None,
        help="comma-separated original trial IDs, e.g. 2,94,100; requires --replay-report",
    )
    parser.add_argument(
        "--map", choices=tuple(item[0] for item in MAPS), action="append",
        help="Select map(s); repeat to run two maps. By default run all three.",
    )
    parser.add_argument(
        "--tf-guarded", action="store_true",
        help="discard a trial with persistent controller TF lag, restart the "
             "stack, and replay that exact scenario",
    )
    parser.add_argument(
        "--max-tf-restarts", type=int, default=1,
        help="maximum stack restarts after a TF fault (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0 or args.timeout_s <= 0:
        parser.error("--trials and --timeout-s must be positive")
    if args.max_tf_restarts < 0:
        parser.error("--max-tf-restarts must be non-negative")
    if (args.arm in ("sac", "sac_only")) != (args.model is not None):
        parser.error("--model is required exactly for --arm sac or sac_only")
    if args.supervisor_params_file is not None and args.arm != "sac_only":
        parser.error("--supervisor-params-file requires --arm sac_only")
    if (args.replay_report is None) != (args.replay_trials is None):
        parser.error("--replay-report and --replay-trials must be used together")
    if args.replay_report is not None and not args.tf_guarded:
        parser.error("replay requires --tf-guarded")
    args.replay_trial_ids = None
    args.replay_scenarios = None
    if args.replay_report is not None:
        if args.map is None or len(args.map) != 1:
            parser.error("replay requires exactly one --map")
        try:
            trial_ids = [int(part.strip()) for part in args.replay_trials.split(",")]
        except ValueError:
            parser.error("--replay-trials must contain integer trial IDs")
        if not trial_ids or any(value <= 0 for value in trial_ids):
            parser.error("--replay-trials must contain positive trial IDs")
        if len(set(trial_ids)) != len(trial_ids):
            parser.error("--replay-trials must not repeat an ID")
        args.replay_report = args.replay_report.expanduser().resolve(strict=True)
        try:
            args.replay_scenarios = load_replay_scenarios(
                args.replay_report, trial_ids, args.map[0],
            )
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid replay report: {error}")
        args.replay_trial_ids = trial_ids
        args.trials = len(trial_ids)
    if args.supervisor_params_file is not None:
        args.supervisor_params_file = args.supervisor_params_file.expanduser().resolve(strict=True)
    model_path = None
    if args.model is not None:
        model_path = args.model.expanduser().resolve(strict=True)
        # Reject an old 54D checkpoint before launching any ROS processes.
        from stable_baselines3 import SAC
        from train_sac import validate_model

        validate_model(
            SAC.load(str(model_path), device="cpu"),
            sac_only=args.arm == "sac_only",
        )

    share = Path(get_package_share_directory("agv_mujoco_sim"))
    output = args.output_dir or Path("benchmark_results") / (
        f"{args.arm}_three_maps_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output = output.resolve()
    if output.exists() and not args.dry_run:
        parser.error(f"output directory already exists: {output}")

    selected_maps = (item for item in MAPS if args.map is None or item[0] in args.map)
    for map_id, model_file, map_file, scenario_profile in selected_maps:
        if not (share / "models" / model_file).is_file():
            raise FileNotFoundError(share / "models" / model_file)
        if not (share / "maps" / map_file).is_file():
            raise FileNotFoundError(share / "maps" / map_file)
        launch_cmd = [
            "ros2", "launch", "agv_mujoco_sim", "phase2_finetune_bringup.launch.py",
            f"model_file:={model_file}",
            f"map_file:={share / 'maps' / map_file}",
            "headless:=true", "real_time_factor:=1.0",
            f"enable_rl_supervisor:={'false' if args.arm == 'pure_mppi' else 'true'}",
            "domain_randomization:=false",
        ]
        if args.arm == "sac_only":
            supervisor_params = args.supervisor_params_file or (
                share / "config" / "rl_supervisor_sac_only.yaml"
            )
            launch_cmd.extend((
                "front_safety_stop_enabled:=false",
                f"supervisor_params_file:={supervisor_params}",
            ))
        bag_cmd = ["ros2", "bag", "record", "-o", str(output / map_id / "rosbag"), *BAG_TOPICS]
        benchmark_cmd = [
            "ros2", "run", "agv_mujoco_sim", "benchmark_dynamic_obstacle.py",
            "--trials", str(args.trials), "--timeout-s", str(args.timeout_s),
            "--output", str(output / map_id / "benchmark.json"),
            "--scenario-profile", scenario_profile,
            "--scenario-seed", str(args.seed),
        ]
        if args.arm == "shield_only":
            benchmark_cmd.append("--shield-only")
        if model_path is not None:
            benchmark_cmd.extend(("--sac-model", str(model_path)))
        if args.arm == "sac_only":
            benchmark_cmd.append("--sac-only")
        if args.dry_run:
            print(f"\n[{map_id}]\n{launch_cmd}\n{bag_cmd}")
            if args.tf_guarded:
                print("TF guarded: one fixed scenario per benchmark process; "
                      "restart and replay only after persistent controller TF lag")
                if args.replay_trial_ids is not None:
                    print("Replay trial IDs and exact scenarios:")
                    for trial_id, scenario in zip(
                        args.replay_trial_ids, args.replay_scenarios
                    ):
                        print(f"  trial {trial_id}: {scenario}")
            else:
                print(benchmark_cmd)
            continue

        map_output = output / map_id
        map_output.mkdir(parents=True)
        if args.tf_guarded:
            run_tf_guarded_map(
                args, map_id, model_file, map_file, scenario_profile,
                launch_cmd, model_path, map_output,
            )
            continue
        print(f"[{map_id}] launching {model_file} with {map_file}", flush=True)
        launch = bag = None
        try:
            for attempt in (1, 2):
                launch_log_path = map_output / (
                    "launch.log" if attempt == 1 else f"launch_retry_{attempt}.log"
                )
                with launch_log_path.open("w") as launch_log:
                    launch = subprocess.Popen(
                        launch_cmd, stdout=launch_log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        vx_max = wait_for_mppi(launch)
                        break
                    except (RuntimeError, TimeoutError) as error:
                        stop_group(launch, "launch")
                        launch = None
                        if attempt == 2:
                            raise RuntimeError(
                                f"Nav2 bringup failed twice; see {launch_log_path}"
                            ) from error
                        print(f"[{map_id}] bringup failed: {error}; retrying", flush=True)
                        time.sleep(3)
            with (map_output / "rosbag.log").open("w") as bag_log, (
                map_output / "benchmark.log"
            ).open("w") as benchmark_log:
                if abs(vx_max - 0.70) > 1e-3:
                    raise RuntimeError(f"Expected MPPI vx_max=0.70 m/s, got {vx_max}")
                if args.arm == "sac_only":
                    require_boolean_parameter(
                        "/rl_velocity_supervisor", "safety_override_enabled", False,
                    )
                    require_boolean_parameter(
                        "/rl_velocity_supervisor", "ttc_observation_enabled", False,
                    )
                    require_boolean_parameter(
                        "/mujoco_ros_bridge", "front_safety_stop_enabled", False,
                    )
                bag = subprocess.Popen(
                    bag_cmd, stdout=bag_log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                time.sleep(3)
                if bag.poll() is not None:
                    raise RuntimeError(f"rosbag exited with code {bag.returncode}")
                print(f"[{map_id}] running {args.trials} trials; output: {map_output}", flush=True)
                result = subprocess.run(
                    benchmark_cmd, stdout=benchmark_log, stderr=subprocess.STDOUT,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Benchmark failed ({result.returncode}); see {map_output / 'benchmark.log'}"
                    )
                report = json.loads((map_output / "benchmark.json").read_text())
                (map_output / "run_metadata.json").write_text(json.dumps({
                    "map_id": map_id, "model_file": model_file, "map_file": map_file,
                    "arm": args.arm, "model": (
                        None if model_path is None else str(model_path)
                    ),
                    "trials": args.trials, "scenario_profile": scenario_profile,
                    "scenario_seed": args.seed, "vx_max_m_s": vx_max,
                    "supervisor_params_file": None if args.supervisor_params_file is None else str(args.supervisor_params_file),
                    "launch_command": launch_cmd, "bag_command": bag_cmd,
                    "benchmark_command": benchmark_cmd,
                    "aggregate": report["aggregate"],
                }, indent=2) + "\n")
                print(f"[{map_id}] {json.dumps(report['aggregate'])}", flush=True)
        finally:
            stop_group(bag, "rosbag")
            stop_group(launch, "launch")
            time.sleep(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
