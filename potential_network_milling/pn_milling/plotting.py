"""Small Pillow-based plotting helper to keep the reproduction dependency-light."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ((31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189))


def _line_plot(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x: np.ndarray, series: list[np.ndarray], labels: list[str], title: str, zero_line: bool = False) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(170, 170, 170), width=1)
    finite_values = np.concatenate([values[np.isfinite(values)] for values in series if np.any(np.isfinite(values))])
    ymin, ymax = float(np.min(finite_values)), float(np.max(finite_values))
    if zero_line:
        ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
    if np.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    padding = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - padding, ymax + padding
    xmin, xmax = float(x[0]), float(x[-1])

    def point(xv: float, yv: float) -> tuple[int, int]:
        px = left + int((xv - xmin) / max(xmax - xmin, 1e-12) * (right - left))
        py = bottom - int((yv - ymin) / (ymax - ymin) * (bottom - top))
        return px, py

    if zero_line and ymin <= 0 <= ymax:
        yzero = point(xmin, 0.0)[1]
        draw.line((left, yzero, right, yzero), fill=(210, 210, 210), width=1)
    for color, values, label in zip(COLORS, series, labels):
        clipped = np.nan_to_num(values, nan=ymax, posinf=ymax, neginf=ymin)
        points = [point(float(xv), float(yv)) for xv, yv in zip(x, clipped)]
        draw.line(points, fill=color, width=3)
    draw.text((left, top - 24), title, fill=(25, 25, 25))
    legend_x = right - 90 * len(labels)
    for idx, (color, label) in enumerate(zip(COLORS, labels)):
        x0 = legend_x + idx * 90
        draw.line((x0, top + 12, x0 + 20, top + 12), fill=color, width=3)
        draw.text((x0 + 25, top + 3), label, fill=(40, 40, 40))
    draw.text((left, bottom + 4), f"{xmin:.2f}", fill=(90, 90, 90))
    draw.text((right - 35, bottom + 4), f"{xmax:.2f}", fill=(90, 90, 90))
    draw.text((left - 58, top), f"{ymax:.2g}", fill=(90, 90, 90))
    draw.text((left - 58, bottom - 12), f"{ymin:.2g}", fill=(90, 90, 90))


def save_overview(path: str | Path, time: np.ndarray, q: np.ndarray, potential: np.ndarray, minimum_h: np.ndarray) -> None:
    image = Image.new("RGB", (1400, 980), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 25), "Potential-network milling reproduction", fill=(20, 20, 20))
    _line_plot(draw, (90, 90, 1340, 350), time, [q[:, 0], q[:, 1], q[:, 2]], ["alpha", "beta", "gamma"], "Attitude coordinates q(t) [rad]", True)
    _line_plot(draw, (90, 410, 1340, 650), time, [potential], ["P"], "Composite potential P(q,p)")
    _line_plot(draw, (90, 710, 1340, 930), time, [minimum_h], ["min H"], "Minimum hard-constraint proximity (feasible above 0)", True)
    image.save(path)

