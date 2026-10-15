#!/usr/bin/env python3
"""Generate matched MuJoCo/Nav2 maps for the three-map MPPI benchmark."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[1]
MAP_DIR = PACKAGE / "maps"
MODEL_DIR = PACKAGE / "models"
TEMPLATE = MODEL_DIR / "rl_trainingworld.xml"
RESOLUTION = 0.05


def rectangles_from_mask(mask: np.ndarray):
    """Merge adjacent occupied pixels into exact axis-aligned rectangles."""
    active = {}
    result = []
    for row_index, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        next_active = {}
        for start, end in zip(starts, ends):
            key = (int(start), int(end))
            first_row = active.pop(key, row_index)
            next_active[key] = first_row
        result.extend((start, end, first, row_index) for (start, end), first in active.items())
        active = next_active
    result.extend((start, end, first, mask.shape[0]) for (start, end), first in active.items())
    return result


def write_yaml(path: Path, image_name: str, origin: tuple[float, float]) -> None:
    path.write_text(
        f"image: {image_name}\nmode: trinary\nresolution: {RESOLUTION}\n"
        f"origin: [{origin[0]}, {origin[1]}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.25\n",
        encoding="utf-8",
    )


def write_world(name: str, image: np.ndarray, origin: tuple[float, float]) -> int:
    root = ET.parse(TEMPLATE).getroot()
    root.set("model", name)
    world = root.find("worldbody")
    assert world is not None
    for child in list(world):
        if child.tag == "body" and child.get("name") == "base_link":
            continue
        world.remove(child)
    light = ET.Element("light", {"directional": "true", "pos": "0 0 10", "dir": "0 0 -1"})
    world.insert(0, light)
    ET.SubElement(world, "geom", {
        "name": "ground_plane", "type": "plane", "size": "100 100 0.1",
        "rgba": "0.82 0.83 0.85 1", "friction": "1.0 0.005 0.0001",
    })
    obstacle = ET.SubElement(world, "body", {
        "name": "dynamic_obstacle", "mocap": "true", "pos": "100 100 0.5",
    })
    ET.SubElement(obstacle, "geom", {
        "name": "dynamic_obstacle_collision", "type": "cylinder",
        "size": "0.45 0.5", "rgba": "0.9 0.1 0.1 1",
    })
    count = 0
    height, _ = image.shape
    for x0, x1, row0, row1 in rectangles_from_mask(image != 254):
        width_m = (x1 - x0) * RESOLUTION
        depth_m = (row1 - row0) * RESOLUTION
        center_x = origin[0] + (x0 + x1) * RESOLUTION / 2
        center_y = origin[1] + (height - (row0 + row1) / 2) * RESOLUTION
        # Each box exactly covers its occupied or unknown raster cells.
        ET.SubElement(world, "geom", {
            "name": f"map_wall_{count}", "type": "box",
            "pos": f"{center_x:.4f} {center_y:.4f} 0.75",
            "size": f"{width_m / 2:.4f} {depth_m / 2:.4f} 0.75",
            "rgba": "0.33 0.35 0.38 1",
        })
        count += 1
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(MODEL_DIR / f"{name}.xml", encoding="utf-8", xml_declaration=True)
    return count


def warehouse(name: str, obstacles: list[tuple[float, float, float, float]]) -> int:
    origin = (-10.0, -10.0)
    image = Image.new("L", (400, 400), 254)
    draw = ImageDraw.Draw(image)
    # Closed outer boundary plus map-specific racks, pallets and crates.
    rects = [
        (-10.0, -10.0, 10.0, -9.8), (-10.0, 9.8, 10.0, 10.0),
        (-10.0, -10.0, -9.8, 10.0), (9.8, -10.0, 10.0, 10.0),
        *obstacles,
    ]
    for xmin, ymin, xmax, ymax in rects:
        left = round((xmin - origin[0]) / RESOLUTION)
        right = round((xmax - origin[0]) / RESOLUTION) - 1
        top = image.height - round((ymax - origin[1]) / RESOLUTION)
        bottom = image.height - round((ymin - origin[1]) / RESOLUTION) - 1
        draw.rectangle((left, top, right, bottom), fill=0)
    array = np.asarray(image)
    image.save(MAP_DIR / f"{name}.pgm")
    write_yaml(MAP_DIR / f"{name}.yaml", f"{name}.pgm", origin)
    return write_world(name, array, origin)


def main() -> None:
    warehouse_a = [
        (-8.0, 2.4, -5.5, 8.0), (-3.5, 2.4, -1.0, 8.0),
        (2.0, 2.4, 4.5, 8.0), (6.0, 2.4, 8.0, 8.0),
        (-8.0, -8.0, -5.5, -2.4), (-3.5, -8.0, -1.0, -2.4),
        (2.0, -8.0, 4.5, -2.4), (6.0, -8.0, 8.0, -2.4),
        (6.4, 0.7, 7.1, 1.4),
    ]
    warehouse_b = [
        (-8.0, 3.0, -6.0, 8.0), (-4.5, 3.0, -2.5, 8.0),
        (1.0, 3.0, 3.0, 8.0), (5.0, 3.0, 8.0, 8.0),
        (-8.0, -8.0, -5.5, -3.0), (-3.5, -8.0, -1.0, -3.0),
        (2.0, -8.0, 4.0, -3.0), (6.0, -8.0, 8.0, -3.0),
        (3.1, 1.0, 3.8, 1.7), (7.3, -1.7, 8.0, -1.0),
    ]
    for name, obstacles in (
        ("mppi_warehouse_aisles", warehouse_a),
        ("mppi_warehouse_mixed", warehouse_b),
    ):
        count = warehouse(name, obstacles)
        print(f"{name}: {count} static boxes")

    source_map = WORKSPACE / "maps" / "floor2" / "floor2_mujoco.pgm"
    corridor_name = "mppi_floor2_corridor"
    shutil.copyfile(source_map, MAP_DIR / f"{corridor_name}.pgm")
    write_yaml(MAP_DIR / f"{corridor_name}.yaml", f"{corridor_name}.pgm", (-4.73, -1.58))
    corridor = np.asarray(Image.open(source_map).convert("L"))
    count = write_world(corridor_name, corridor, (-4.73, -1.58))
    print(f"{corridor_name}: {count} static boxes from floor2 map")


if __name__ == "__main__":
    main()
