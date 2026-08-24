import pytest

from vlm_sim.autonomy import (
    POLICY_OUTPUT_CONTRACT,
    SUPPORTED_ACTIONS,
    build_policy_prompt,
    parse_policy_decision,
    should_stop,
)


def test_parse_fenced_policy_and_threshold_stop():
    decision = parse_policy_decision(
        """```json
        {"observation":"mug visible","new_information":"target found","action":"Stop",
         "confidence":0.92,"task_status":"completed","rationale":"clear view"}
        ```"""
    )
    assert decision.action == "Stop"
    assert decision.confidence == 0.92
    assert should_stop(decision, 0.85) == "requested confidence threshold reached"


def test_prompt_contains_task_and_memory():
    decision = parse_policy_decision(
        '{"observation":"door","new_information":"","action":"RotateRight",'
        '"confidence":0.3,"task_status":"exploring","rationale":"scan"}'
    )
    prompt = build_policy_prompt("find a mug", [decision], 0.8)
    assert "find a mug" in prompt
    assert "RotateRight" in prompt
    assert "current egocentric RGB observation" in prompt
    assert POLICY_OUTPUT_CONTRACT in prompt
    assert all(action in prompt for action in SUPPORTED_ACTIONS)


def test_rejects_unsupported_action():
    with pytest.raises(ValueError, match="Unsupported VLM action"):
        parse_policy_decision('{"action":"Teleport","confidence":0.5}')


def test_confidence_threshold_stops_even_before_stop_action():
    decision = parse_policy_decision(
        '{"action":"RotateLeft","confidence":0.9,"task_status":"exploring"}'
    )
    assert should_stop(decision, 0.85) == "requested confidence threshold reached"


def test_native_done_stops_policy():
    decision = parse_policy_decision(
        '{"action":"Done","confidence":0.4,"task_status":"exploring"}'
    )
    assert should_stop(decision, 0.85) == "policy selected AI2-THOR Done"
