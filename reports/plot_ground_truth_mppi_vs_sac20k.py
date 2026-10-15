"""Plot MuJoCo body-frame ground-truth forward speed from saved rosbags."""

import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message


ROOT = Path(__file__).resolve().parents[1] / "benchmark_results"
RUNS = {
    "MPPI thuần": ROOT / "mppi_three_maps_20261011_235102",
    "MPPI + SAC 20k": ROOT / "sac_only_conservative_two_warehouse_seed42_retry",
}


def ground_truth_by_position(run, map_name):
    bag = RUNS[run] / map_name / "rosbag" / "rosbag_0.db3"
    conn = sqlite3.connect(f"file:{bag}?mode=ro", uri=True)
    topics = dict(conn.execute("SELECT name, id FROM topics"))
    odom_id = topics["/ground_truth/odom"]
    obstacle_id = topics["/dynamic_obstacle/position"]
    readings = defaultdict(list)
    obstacle_active = False
    ready_after_reset = True
    query = ("SELECT topic_id, data FROM messages WHERE topic_id IN (?, ?) "
             "ORDER BY timestamp")
    for topic_id, data in conn.execute(query, (odom_id, obstacle_id)):
        if topic_id == obstacle_id:
            point = deserialize_message(data, Point)
            if point.x < 10.0:
                obstacle_active = True
            else:
                if obstacle_active:
                    ready_after_reset = False
                obstacle_active = False
            continue
        odom = deserialize_message(data, Odometry)
        x = odom.pose.pose.position.x
        if x < 0.2 and not obstacle_active:
            ready_after_reset = True
        if not ready_after_reset or obstacle_active or not (0.5 <= x < 2.5):
            continue
        index = min(19, int((x - 0.5) / 0.1))
        readings[index].append(float(odom.twist.twist.linear.x))
    conn.close()
    xs = [round(0.55 + 0.1 * index, 2) for index in range(20)]
    ys = [mean(readings[index]) if readings[index] else float("nan")
          for index in range(20)]
    return xs, ys, sum(map(len, readings.values()))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for ax, map_name, title in zip(
        axes,
        ("mppi_warehouse_aisles", "mppi_warehouse_mixed"),
        ("Warehouse aisles", "Warehouse mixed"),
    ):
        for run, color in (("MPPI thuần", "#334155"),
                           ("MPPI + SAC 20k", "#d35432")):
            xs, ys, count = ground_truth_by_position(run, map_name)
            ax.plot(xs, ys, color=color, linewidth=3, label=run)
            print(f"{map_name} {run}: {count} ground-truth samples")
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 0.78)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Vị trí x của robot (m)")
    axes[0].set_ylabel("Vận tốc thân robot từ MuJoCo (m/s)")
    fig.suptitle("Vận tốc ground truth trước khi vật cản động kích hoạt", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.91), w_pad=2.8)
    output = Path(__file__).with_name("ground_truth_speed_mppi_vs_sac20k_two_warehouse.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
