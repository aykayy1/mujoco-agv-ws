"""Plot saved 20k SAC and paired MPPI benchmark samples (no simulation)."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1] / "benchmark_results"
BENCHMARKS = {
    "MPPI": ROOT / "mppi_three_maps_20261011_235102",
    "SAC": ROOT / "sac_only_conservative_two_warehouse_seed42_retry",
}


def series(mode, map_name):
    path = BENCHMARKS[mode] / map_name
    meta = json.loads((path / "benchmark.json").read_text())
    trigger = {int(t["trial"]): t["scenario"]["trigger_x"] for t in meta["trials"]}
    trial_bins = defaultdict(list)
    with (path / "benchmark.samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            trial, x = int(row["trial"]), float(row["x"])
            if not (0.5 <= x < 2.5 and x < trigger[trial]
                    and row["obstacle_active"] == "False"):
                continue
            bin_index = min(19, int((x - 0.5) / 0.1))
            values = {key: float(row[key]) for key in
                      ("mppi_cmd_vx", "final_cmd_vx", "actual_vx") if row[key]}
            if mode == "SAC":
                values["requested_cap"] = 0.70 * float(row["sac_requested_scale"])
                values["applied_cap"] = 0.70 * float(row["supervisor_applied_scale"])
            trial_bins[(trial, bin_index)].append(values)

    bins = defaultdict(lambda: defaultdict(list))
    for (_, bin_index), rows in trial_bins.items():
        for key in rows[0]:
            vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
            if vals:
                bins[bin_index][key].append(statistics.mean(vals))
    keys = {key for bucket in bins.values() for key in bucket}
    xs = [0.55 + index * 0.1 for index in range(20)]
    return {key: (xs, [statistics.mean(bins[index][key])
                      if bins[index][key] else math.nan
                      for index in range(20)]) for key in keys}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, sharey=True)
    colors = {
        "baseline": "#444b5a", "requested": "#bc5c26",
        "applied": "#dc9a19", "mppi": "#4b73b7",
        "final": "#8754a6", "actual": "#23846b",
    }
    for col, name in enumerate(("mppi_warehouse_aisles", "mppi_warehouse_mixed")):
        baseline, sac = series("MPPI", name), series("SAC", name)
        upper, lower = axes[:, col]
        upper.plot(*baseline["mppi_cmd_vx"], label="MPPI thuần: cmd vx",
                   color=colors["baseline"], linewidth=2.2)
        upper.plot(*sac["requested_cap"], label="SAC yêu cầu × 0.70",
                   color=colors["requested"], linestyle="--", linewidth=1.9)
        upper.plot(*sac["applied_cap"], label="Giới hạn áp dụng × 0.70",
                   color=colors["applied"], linewidth=2)
        upper.plot(*sac["mppi_cmd_vx"], label="MPPI + SAC: cmd vx",
                   color=colors["mppi"], linewidth=2)
        lower.plot(*sac["mppi_cmd_vx"], label="MPPI cmd vx",
                   color=colors["mppi"], linewidth=2.5)
        lower.plot(*sac["final_cmd_vx"], label="Final cmd vx",
                   color=colors["final"], linestyle="--", linewidth=1.8)
        lower.plot(*sac["actual_vx"], label="Odometry vx",
                   color=colors["actual"], linestyle=":", linewidth=1.5)
        upper.set_title("Aisles" if col == 0 else "Mixed", fontweight="semibold")
        lower.set_xlabel("Vị trí x trước khi vật cản động kích hoạt (m)")
        for ax in (upper, lower):
            ax.set_xlim(0.5, 2.5)
            ax.set_ylim(0, 0.78)
            ax.set_xticks([0.5, 1, 1.5, 2, 2.5])
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Giới hạn và lệnh vx (m/s)")
    axes[1, 0].set_ylabel("Lệnh và tốc độ thực (m/s)")
    fig.suptitle("SAC 20k giảm cmd_vel qua SpeedLimit của Nav2",
                 fontsize=15, fontweight="semibold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False, fontsize=9)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.90), h_pad=2.7, w_pad=2)
    output = Path(__file__).with_name("speed_scale_20k_cmd_vel_two_warehouse.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
