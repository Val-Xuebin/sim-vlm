from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Observation:
    image: Image.Image
    metadata: dict[str, Any]
    depth: Any | None = None
    instance_segmentation: Image.Image | None = None
    semantic_segmentation: Image.Image | None = None
    instance_detections2d: dict[str, Any] | None = None


class ThorSimulator:
    """Small lifecycle-safe wrapper around an AI2-THOR controller."""

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 640,
        height: int = 480,
        render_depth: bool = False,
        render_instance_segmentation: bool = False,
        render_semantic_segmentation: bool = False,
    ):
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering, Linux64

        platform_name = os.environ.get("AI2THOR_PLATFORM", "Linux64")
        platforms = {"CloudRendering": CloudRendering, "Linux64": Linux64}
        if platform_name not in platforms:
            raise ValueError(f"Unsupported AI2THOR_PLATFORM={platform_name!r}")

        self.controller = Controller(
            scene=scene,
            width=width,
            height=height,
            platform=platforms[platform_name],
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_segmentation,
            renderSemanticSegmentation=render_semantic_segmentation,
        )

    @staticmethod
    def _observation(event: Any) -> Observation:
        instance = getattr(event, "instance_segmentation_frame", None)
        semantic = getattr(event, "class_segmentation_frame", None)
        return Observation(
            image=Image.fromarray(event.frame),
            metadata=event.metadata,
            depth=getattr(event, "depth_frame", None),
            instance_segmentation=Image.fromarray(instance) if instance is not None else None,
            semantic_segmentation=Image.fromarray(semantic) if semantic is not None else None,
            instance_detections2d=getattr(event, "instance_detections2D", None),
        )

    def observe(self) -> Observation:
        return self._observation(self.controller.last_event)

    def step(self, action: str, raise_on_failure: bool = True, **kwargs: Any) -> Observation:
        event = self.controller.step(action=action, **kwargs)
        if raise_on_failure and not event.metadata.get("lastActionSuccess", False):
            error = event.metadata.get("errorMessage", "unknown AI2-THOR error")
            raise RuntimeError(f"{action} failed: {error}")
        return self._observation(event)

    def close(self) -> None:
        self.controller.stop()

    def __enter__(self) -> "ThorSimulator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    agent = metadata.get("agent", {})
    visible = []
    for obj in metadata.get("objects", []):
        if obj.get("visible"):
            visible.append(
                {
                    "objectId": obj.get("objectId"),
                    "objectType": obj.get("objectType"),
                    "distance": obj.get("distance"),
                    "pickupable": obj.get("pickupable"),
                    "openable": obj.get("openable"),
                    "isOpen": obj.get("isOpen"),
                }
            )
    return {
        "sceneName": metadata.get("sceneName"),
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "agent": {"position": agent.get("position"), "rotation": agent.get("rotation")},
        "visibleObjects": visible,
    }


def save_observation(observation: Observation, output_dir: Path) -> tuple[Path, Path]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "frame.png"
    metadata_path = output_dir / "metadata.json"
    observation.image.save(image_path)
    metadata_path.write_text(
        json.dumps(compact_metadata(observation.metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return image_path, metadata_path
