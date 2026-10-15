#!/usr/bin/env python3
"""Recompute moving-encounter fields in a completed benchmark from its CSV."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from benchmark_dynamic_obstacle import aggregate_trials, moving_encounter_metrics


def csv_rows(path: Path) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row["obstacle_active"] = row["obstacle_active"] == "True"
            row["obstacle_moving"] = row["obstacle_moving"] == "True"
            for key in (
                "obstacle_center_distance_m", "estimated_body_clearance_m",
                "elapsed_sim_s", "obstacle_y",
            ):
                row[key] = None if row[key] == "" else float(row[key])
            grouped[int(row["trial"])].append(row)
    return grouped


def write_with_backup(path: Path, data: dict) -> None:
    backup = path.with_name(path.stem + ".before_moving_fix.json")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def repair_report(path: Path, rows_by_trial: dict[int, list[dict]],
                  *, original_trial_id: int | None = None) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    for trial in report["trials"]:
        trial_id = original_trial_id if original_trial_id is not None else int(trial["trial"])
        if trial_id not in rows_by_trial:
            raise ValueError(f"Missing samples for trial {trial_id} in {path}")
        trial.update(moving_encounter_metrics(rows_by_trial[trial_id]))
    report["aggregate"] = aggregate_trials(report["trials"])
    write_with_backup(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    grouped = csv_rows(run_dir / "benchmark.samples.csv")
    trial_reports = sorted(run_dir.glob("attempt_*/trial_*/benchmark.json"))
    for trial_path in trial_reports:
        trial_id = int(trial_path.parent.name.removeprefix("trial_"))
        report = repair_report(trial_path, grouped, original_trial_id=trial_id)
        progress_path = trial_path.parent / "benchmark.progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress["trials"] = report["trials"]
            progress["aggregate_so_far"] = report["aggregate"]
            write_with_backup(progress_path, progress)
    combined = repair_report(run_dir / "benchmark.json", grouped)
    progress_path = run_dir / "benchmark.progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["trials"] = combined["trials"]
        progress["aggregate_so_far"] = combined["aggregate"]
        write_with_backup(progress_path, progress)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["aggregate"] = combined["aggregate"]
        write_with_backup(metadata_path, metadata)
    print(json.dumps(combined["aggregate"], indent=2))


if __name__ == "__main__":
    main()
