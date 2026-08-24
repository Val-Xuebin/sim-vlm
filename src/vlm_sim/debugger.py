from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .simulator import Observation, ThorSimulator, compact_metadata


@dataclass
class HistoryEntry:
    step: int
    action: str
    success: bool
    elapsed_ms: float
    position: dict[str, float]
    rotation_y: float
    thumbnail: Image.Image


class SimulatorDebugger:
    """Persistent, single-user AI2-THOR session for the local debugger UI."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.simulator: ThorSimulator | None = None
        self.observation: Observation | None = None
        self.reachable_positions: list[dict[str, float]] = []
        self.trajectory: list[dict[str, float]] = []
        self.history: list[HistoryEntry] = []

    def reset(self, scene: str, enable_segmentation: bool = False) -> Observation:
        with self._lock:
            self.close()
            self.simulator = ThorSimulator(
                scene=scene,
                width=int(os.environ.get("AI2THOR_DEBUG_WIDTH", "320")),
                height=int(os.environ.get("AI2THOR_DEBUG_HEIGHT", "240")),
                render_depth=False,
                render_instance_segmentation=enable_segmentation,
            )
            self.observation = self.simulator.observe()
            reachable = self.simulator.controller.step(action="GetReachablePositions")
            self.reachable_positions = reachable.metadata.get("actionReturn") or []
            position = self.observation.metadata.get("agent", {}).get("position", {}).copy()
            self.trajectory = [position]
            self.history = []
            self._append_history("Reset", 0.0)
            return self.observation

    def step(self, action: str, **kwargs: Any) -> Observation:
        with self._lock:
            if self.simulator is None:
                raise RuntimeError("Reset a scene before taking an action.")
            started = time.perf_counter()
            self.observation = self.simulator.step(
                action, raise_on_failure=False, **kwargs
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            position = self.observation.metadata.get("agent", {}).get("position", {}).copy()
            self.trajectory.append(position)
            self._append_history(action, elapsed_ms)
            return self.observation

    def _append_history(self, action: str, elapsed_ms: float) -> None:
        assert self.observation is not None
        agent = self.observation.metadata.get("agent", {})
        self.history.append(
            HistoryEntry(
                step=len(self.history),
                action=action,
                success=bool(self.observation.metadata.get("lastActionSuccess", True)),
                elapsed_ms=elapsed_ms,
                position=agent.get("position", {}).copy(),
                rotation_y=float(agent.get("rotation", {}).get("y", 0.0)),
                thumbnail=self.observation.image.copy().resize((160, 120)),
            )
        )

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None

    def analysis_snapshot(self) -> tuple[Image.Image, dict[str, Any], int]:
        """Return an immutable VLM input tied to the current debugger step."""
        with self._lock:
            if self.observation is None:
                raise RuntimeError("Reset a scene before analyzing an observation.")
            return (
                self.observation.image.copy(),
                compact_metadata(self.observation.metadata),
                max(0, len(self.history) - 1),
            )

    def visible_objects(self) -> list[dict[str, Any]]:
        if self.observation is None:
            return []
        return sorted(
            [obj for obj in self.observation.metadata.get("objects", []) if obj.get("visible")],
            key=lambda obj: (obj.get("objectType", ""), obj.get("distance", math.inf)),
        )

    def sensor_image(self, sensor: str, overlays: list[str] | None = None) -> Image.Image:
        if self.observation is None:
            return Image.new("RGB", (640, 480), "#111827")
        observation = self.observation
        if sensor == "Depth" and observation.depth is not None:
            depth = np.asarray(observation.depth, dtype=np.float32)
            valid = depth[np.isfinite(depth)]
            ceiling = float(np.percentile(valid, 95)) if valid.size else 1.0
            normalized = np.clip(depth / max(ceiling, 1e-6), 0, 1)
            image = Image.fromarray(np.uint8((1.0 - normalized) * 255), mode="L").convert("RGB")
        elif sensor == "Instance Seg" and observation.instance_segmentation is not None:
            image = observation.instance_segmentation.copy()
        elif sensor == "Instance Seg":
            image = Image.new("RGB", observation.image.size, "#111827")
            draw = ImageDraw.Draw(image)
            draw.text(
                (18, 18),
                "Instance segmentation is disabled.\nEnable it before Reset Scene.\n"
                "Software rendering may be slow.",
                fill="#e2e8f0",
                spacing=8,
            )
        elif sensor == "Semantic Seg" and observation.semantic_segmentation is not None:
            image = observation.semantic_segmentation.copy()
        else:
            image = observation.image.copy()

        overlays = overlays or []
        if sensor == "RGB" and observation.instance_detections2d:
            draw = ImageDraw.Draw(image)
            by_id = {obj.get("objectId"): obj for obj in self.visible_objects()}
            for object_id, box in observation.instance_detections2d.items():
                if object_id not in by_id:
                    continue
                x1, y1, x2, y2 = [int(value) for value in box]
                if "Bounding Boxes" in overlays:
                    draw.rectangle((x1, y1, x2, y2), outline="#22d3ee", width=2)
                if "Object Labels" in overlays:
                    label = str(by_id[object_id].get("objectType", "Object"))
                    draw.rectangle((x1, max(0, y1 - 16), x1 + 7 * len(label) + 6, y1), fill="#0891b2")
                    draw.text((x1 + 3, max(0, y1 - 15)), label, fill="white")
        return image

    def top_down_map(
        self,
        show_reachable: bool = True,
        show_objects: bool = True,
        show_path: bool = True,
    ) -> Image.Image:
        canvas = Image.new("RGB", (520, 360), "#0f172a")
        draw = ImageDraw.Draw(canvas)
        points = list(self.reachable_positions) + list(self.trajectory)
        if self.observation is not None and show_objects:
            points += [obj.get("position", {}) for obj in self.visible_objects()]
        xs = [float(p.get("x", 0)) for p in points] or [-1, 1]
        zs = [float(p.get("z", 0)) for p in points] or [-1, 1]
        margin = 28
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        span_x, span_z = max(max_x - min_x, 1), max(max_z - min_z, 1)

        def project(point: dict[str, float]) -> tuple[int, int]:
            x = margin + (float(point.get("x", 0)) - min_x) / span_x * (canvas.width - 2 * margin)
            y = canvas.height - margin - (float(point.get("z", 0)) - min_z) / span_z * (
                canvas.height - 2 * margin
            )
            return int(x), int(y)

        if show_reachable:
            for point in self.reachable_positions:
                x, y = project(point)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#475569")
        if show_path and len(self.trajectory) > 1:
            draw.line([project(point) for point in self.trajectory], fill="#38bdf8", width=3)
        if show_objects and self.observation is not None:
            for obj in self.visible_objects():
                x, y = project(obj.get("position", {}))
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#f59e0b")
                draw.text((x + 6, y - 7), str(obj.get("objectType", "")), fill="#fbbf24")
        if self.observation is not None:
            agent = self.observation.metadata.get("agent", {})
            position = agent.get("position", {})
            x, y = project(position)
            yaw = math.radians(float(agent.get("rotation", {}).get("y", 0)))
            dx, dy = 14 * math.sin(yaw), -14 * math.cos(yaw)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#22c55e")
            draw.line((x, y, x + dx, y + dy), fill="white", width=3)
        return canvas

    def history_rows(self) -> list[list[Any]]:
        return [
            [
                item.step,
                item.action,
                item.success,
                round(item.elapsed_ms, 1),
                round(float(item.position.get("x", 0)), 2),
                round(float(item.position.get("z", 0)), 2),
                round(item.rotation_y, 1),
            ]
            for item in self.history
        ]

    def history_gallery(self) -> list[tuple[Image.Image, str]]:
        return [(item.thumbnail, f"t={item.step} · {item.action}") for item in self.history]
