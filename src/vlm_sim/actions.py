from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    category: str
    requires_target: bool = False
    capability: str | None = None


ACTION_SPECS: dict[str, ActionSpec] = {
    # Agent navigation and posture.
    "MoveAhead": ActionSpec("Navigation"),
    "MoveBack": ActionSpec("Navigation"),
    "MoveLeft": ActionSpec("Navigation"),
    "MoveRight": ActionSpec("Navigation"),
    "RotateLeft": ActionSpec("Navigation"),
    "RotateRight": ActionSpec("Navigation"),
    "LookUp": ActionSpec("Navigation"),
    "LookDown": ActionSpec("Navigation"),
    "Crouch": ActionSpec("Navigation"),
    "Stand": ActionSpec("Navigation"),
    "Done": ActionSpec("Navigation"),
    # Inventory and physical movement.
    "PickupObject": ActionSpec("Inventory", True, "pickupable"),
    "PutObject": ActionSpec("Inventory", True, "receptacle"),
    "DropHandObject": ActionSpec("Inventory"),
    "ThrowObject": ActionSpec("Inventory"),
    "PushObject": ActionSpec("Object movement", True, "moveable"),
    "PullObject": ActionSpec("Object movement", True, "moveable"),
    # Object state changes exposed by iTHOR.
    "OpenObject": ActionSpec("Object state", True, "openable"),
    "CloseObject": ActionSpec("Object state", True, "openable"),
    "ToggleObjectOn": ActionSpec("Object state", True, "toggleable"),
    "ToggleObjectOff": ActionSpec("Object state", True, "toggleable"),
    "BreakObject": ActionSpec("Object state", True, "breakable"),
    "SliceObject": ActionSpec("Object state", True, "sliceable"),
    "CookObject": ActionSpec("Object state", True, "cookable"),
    "DirtyObject": ActionSpec("Object state", True, "dirtyable"),
    "CleanObject": ActionSpec("Object state", True, "dirtyable"),
    "FillObjectWithLiquid": ActionSpec("Object state", True, "canFillWithLiquid"),
    "EmptyLiquidFromObject": ActionSpec("Object state", True, "canFillWithLiquid"),
    "UseUpObject": ActionSpec("Object state", True, "canBeUsedUp"),
}

POLICY_ACTIONS = tuple(ACTION_SPECS) + ("Stop",)
TARGET_ACTIONS = {name for name, spec in ACTION_SPECS.items() if spec.requires_target}


def held_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((obj for obj in objects if obj.get("isPickedUp")), None)


def action_available(
    action: str, target: dict[str, Any] | None, objects: list[dict[str, Any]]
) -> bool:
    spec = ACTION_SPECS[action]
    held = held_object(objects)
    if action in {"DropHandObject", "ThrowObject"}:
        return held is not None
    if action == "PutObject":
        return held is not None and bool(target and target.get("receptacle"))
    if spec.requires_target and target is None:
        return False
    if spec.capability and not bool(target and target.get(spec.capability)):
        return False
    if action == "PickupObject":
        return held is None and not bool(target and target.get("isPickedUp"))
    if action == "OpenObject":
        return not bool(target and target.get("isOpen"))
    if action == "CloseObject":
        return bool(target and target.get("isOpen"))
    if action == "ToggleObjectOn":
        return not bool(target and target.get("isToggled"))
    if action == "ToggleObjectOff":
        return bool(target and target.get("isToggled"))
    if action == "BreakObject":
        return not bool(target and target.get("isBroken"))
    if action == "SliceObject":
        return (
            not bool(target and target.get("isSliced"))
            and held is not None
            and held.get("objectType") in {"Knife", "ButterKnife"}
        )
    if action == "CookObject":
        return not bool(target and target.get("isCooked"))
    if action == "DirtyObject":
        return not bool(target and target.get("isDirty"))
    if action == "CleanObject":
        return bool(target and target.get("isDirty"))
    if action == "FillObjectWithLiquid":
        return not bool(target and target.get("isFilledWithLiquid"))
    if action == "EmptyLiquidFromObject":
        return bool(target and target.get("isFilledWithLiquid"))
    if action == "UseUpObject":
        return not bool(target and target.get("isUsedUp"))
    return True


def resolve_target(
    target: str | None, objects: list[dict[str, Any]], action: str
) -> dict[str, Any]:
    visible = [obj for obj in objects if obj.get("visible")]
    exact = next((obj for obj in visible if obj.get("objectId") == target), None)
    candidates = [
        obj
        for obj in visible
        if target and str(obj.get("objectType", "")).lower() == target.lower()
    ]
    candidates.sort(key=lambda obj: float(obj.get("distance", float("inf"))))
    resolved = exact or (candidates[0] if candidates else None)
    if resolved is None:
        raise ValueError(f"No visible target matches {target!r}")
    if not action_available(action, resolved, objects):
        raise ValueError(f"{action} is not currently valid for {resolved.get('objectType')}")
    return resolved


def action_kwargs(
    action: str,
    target: str | None,
    parameters: dict[str, Any] | None,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    if action not in ACTION_SPECS:
        raise ValueError(f"Unsupported simulator action: {action}")
    parameters = parameters or {}
    spec = ACTION_SPECS[action]
    kwargs: dict[str, Any] = {}
    if spec.requires_target:
        resolved = resolve_target(target, objects, action)
        kwargs["objectId"] = resolved["objectId"]
    elif not action_available(action, None, objects):
        raise ValueError(f"{action} is not currently valid")

    if action == "OpenObject":
        kwargs["openness"] = min(1.0, max(0.0, float(parameters.get("openness", 1.0))))
    if action in {"PushObject", "PullObject", "ThrowObject"}:
        kwargs["moveMagnitude"] = min(
            2.0, max(0.05, float(parameters.get("moveMagnitude", 0.25)))
        )
    if action == "FillObjectWithLiquid":
        liquid = str(parameters.get("fillLiquid", "water")).lower()
        if liquid not in {"water", "coffee", "wine"}:
            raise ValueError("fillLiquid must be water, coffee, or wine")
        kwargs["fillLiquid"] = liquid
    return kwargs
