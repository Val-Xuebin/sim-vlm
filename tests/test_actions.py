import pytest

from vlm_sim.actions import action_available, action_kwargs, resolve_target

OBJECTS = [
    {
        "objectId": "Mug|1",
        "objectType": "Mug",
        "visible": True,
        "distance": 0.8,
        "pickupable": True,
        "canFillWithLiquid": True,
        "isFilledWithLiquid": False,
    },
    {
        "objectId": "Cabinet|1",
        "objectType": "Cabinet",
        "visible": True,
        "distance": 1.0,
        "openable": True,
        "isOpen": False,
        "receptacle": True,
    },
]


def test_resolves_visual_type_to_visible_object_id():
    target = resolve_target("Mug", OBJECTS, "PickupObject")
    assert target["objectId"] == "Mug|1"
    assert action_kwargs("PickupObject", "Mug", {}, OBJECTS) == {"objectId": "Mug|1"}


def test_state_buttons_follow_object_state():
    cabinet = OBJECTS[1]
    assert action_available("OpenObject", cabinet, OBJECTS)
    assert not action_available("CloseObject", cabinet, OBJECTS)


def test_rejects_hallucinated_or_invalid_target():
    with pytest.raises(ValueError, match="No visible target"):
        action_kwargs("OpenObject", "Drawer", {}, OBJECTS)
    with pytest.raises(ValueError, match="not currently valid"):
        action_kwargs("OpenObject", "Mug", {}, OBJECTS)


def test_sanitizes_parameters():
    assert action_kwargs(
        "FillObjectWithLiquid", "Mug", {"fillLiquid": "coffee"}, OBJECTS
    ) == {"objectId": "Mug|1", "fillLiquid": "coffee"}
    with pytest.raises(ValueError, match="fillLiquid"):
        action_kwargs("FillObjectWithLiquid", "Mug", {"fillLiquid": "oil"}, OBJECTS)
