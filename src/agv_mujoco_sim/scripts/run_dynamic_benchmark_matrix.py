#!/usr/bin/env python3
"""Run a reproducible dynamic-obstacle benchmark matrix on an active stack."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ament_index_python.packages import get_package_share_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected controller arms over dynamic scenarios."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / "dynamic_matrix",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paired-domain-randomization",
        action="store_true",
        help="restart the same deterministic randomization sequence for every cell",
    )
    return parser.parse_args()


def cli_args(values: Dict[str, Any]) -> Iterable[str]:
    for name, value in values.items():
        option = f"--{name}"
        if isinstance(value, bool):
            if value:
                yield option
        else:
            yield option
            yield str(value)


def main() -> int:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    share = Path(get_package_share_directory("agv_mujoco_sim"))
    config_path = args.config or share / "config" / "dynamic_benchmark_scenarios.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arms = dict(config["arms"])
    scenarios = list(config["scenarios"])
    selected_arms = args.arm or list(arms)
    selected_scenarios = args.scenario or [item["id"] for item in scenarios]

    unknown_arms = sorted(set(selected_arms) - set(arms))
    scenario_by_id = {item["id"]: item for item in scenarios}
    unknown_scenarios = sorted(set(selected_scenarios) - set(scenario_by_id))
    if unknown_arms:
        raise ValueError(f"Unknown arms: {unknown_arms}")
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}")

    benchmark = Path(__file__).with_name("benchmark_dynamic_obstacle.py")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    for scenario_id in selected_scenarios:
        scenario = scenario_by_id[scenario_id]
        for arm in selected_arms:
            output = output_dir / f"{scenario_id}__{arm}.json"
            command = [
                sys.executable,
                str(benchmark),
                "--trials",
                str(args.trials),
                "--output",
                str(output),
                *cli_args(dict(scenario.get("args", {}))),
                *cli_args(dict(arms[arm])),
            ]
            if args.paired_domain_randomization:
                command.append("--paired-domain-randomization")
            print(f"[matrix] {scenario_id} / {arm}", flush=True)
            print(" ".join(command), flush=True)
            record: Dict[str, Any] = {
                "scenario": scenario_id,
                "arm": arm,
                "output": str(output),
                "command": command,
                "status": "dry_run" if args.dry_run else "running",
            }
            manifest.append(record)
            if not args.dry_run:
                completed = subprocess.run(command, check=False)
                record["returncode"] = completed.returncode
                record["status"] = (
                    "completed" if completed.returncode == 0 else "failed"
                )
                if completed.returncode == 0:
                    report = json.loads(output.read_text(encoding="utf-8"))
                    record["aggregate"] = report["aggregate"]
                if completed.returncode != 0:
                    break
        if manifest[-1]["status"] == "failed":
            break

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                "config": str(config_path.resolve()),
                "trials_per_cell": args.trials,
                "paired_domain_randomization": (
                    args.paired_domain_randomization
                ),
                "runs": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.csv"
    summary_fields = (
        "scenario",
        "arm",
        "status",
        "trial_count",
        "success_rate",
        "collision_count",
        "mean_success_time_s",
        "mean_braking_onset_delay_s",
        "mean_deceleration_delay_s",
        "minimum_estimated_body_clearance_m",
        "output",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for item in manifest:
            aggregate = item.get("aggregate", {})
            writer.writerow(
                {
                    "scenario": item["scenario"],
                    "arm": item["arm"],
                    "status": item["status"],
                    "trial_count": aggregate.get("trial_count"),
                    "success_rate": aggregate.get("success_rate"),
                    "collision_count": aggregate.get("collision_count"),
                    "mean_success_time_s": aggregate.get(
                        "mean_success_time_s"
                    ),
                    "mean_braking_onset_delay_s": aggregate.get(
                        "mean_braking_onset_delay_s"
                    ),
                    "mean_deceleration_delay_s": aggregate.get(
                        "mean_deceleration_delay_s"
                    ),
                    "minimum_estimated_body_clearance_m": aggregate.get(
                        "minimum_estimated_body_clearance_m"
                    ),
                    "output": item["output"],
                }
            )
    print(f"[matrix] manifest: {manifest_path}", flush=True)
    print(f"[matrix] summary: {summary_path}", flush=True)
    return 1 if any(item["status"] == "failed" for item in manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
