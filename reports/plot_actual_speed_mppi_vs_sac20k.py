"""Compare measured robot speed from paired MPPI and SAC 20k CSV runs."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_speed_scale_20k_cmd_vel import series


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for ax, map_name, title in zip(
        axes,
        ("mppi_warehouse_aisles", "mppi_warehouse_mixed"),
        ("Warehouse aisles", "Warehouse mixed"),
    ):
        mppi = series("MPPI", map_name)["actual_vx"]
        sac = series("SAC", map_name)["actual_vx"]
        ax.plot(*mppi, color="#334155", linewidth=3, label="Không SAC (MPPI thuần)")
        ax.plot(*sac, color="#d35432", linewidth=3, label="Có SAC 20k")
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 0.78)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Vị trí x của robot (m)")
    axes[0].set_ylabel("Tốc độ thực đo bằng odometry (m/s)")
    fig.suptitle("Tốc độ thực của robot trước khi vật cản động kích hoạt", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.91), w_pad=2.8)
    output = Path(__file__).with_name("actual_speed_mppi_vs_sac20k_two_warehouse.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
