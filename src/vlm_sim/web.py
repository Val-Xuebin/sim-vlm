from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Direct UI launches must use the same project cache as CLI launchers.
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECT_HF_HOME = PROJECT_DIR / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(PROJECT_HF_HOME))
if (PROJECT_HF_HOME / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .backends import VLMBackend, make_backend
from .debugger import SimulatorDebugger


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_OPTIONS = {
    "metadata": ["simulator-metadata"],
    "transformers": [
        DEFAULT_MODEL,
        "Qwen/Qwen3-VL-4B-Instruct",
    ],
    "openai": ["gpt-4.1-mini"],
}
BACKEND_OPTIONS = ["metadata", "transformers", "openai"]


def load_model_options() -> dict[str, list[str]]:
    config_path = Path(
        os.environ.get("VLM_SIM_MODEL_CONFIG", PROJECT_DIR / "configs" / "models.json")
    )
    if not config_path.is_file():
        return DEFAULT_MODEL_OPTIONS
    configured = json.loads(config_path.read_text(encoding="utf-8"))
    options = {name: configured.get(name, values) for name, values in DEFAULT_MODEL_OPTIONS.items()}
    if any(not isinstance(values, list) or not values for values in options.values()):
        raise ValueError(f"Every backend in {config_path} must contain a non-empty model list")
    return options
DEFAULT_PROMPT = (
    "Analyze this embodied-agent observation. Return concise JSON with keys: "
    "scene_summary, visible_objects, spatial_relations, hazards, and recommended_action. "
    "recommended_action must be one of MoveAhead, MoveBack, MoveLeft, MoveRight, "
    "RotateLeft, RotateRight, LookUp, LookDown, Crouch, Stand, or Stop."
)
PROMPTS = {
    "Scene understanding": DEFAULT_PROMPT,
    "Navigation assistant": (
        "Describe free space and obstacles, then recommend exactly one safe navigation action "
        "from MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, or Stop. "
        "Explain the visual evidence briefly."
    ),
    "Object inventory": (
        "List the important visible objects with approximate image location (left, center, right) "
        "and likely affordances. Do not infer hidden objects."
    ),
    "Safety check": (
        "Inspect this robot observation for collision risks, hazards, blocked paths, and uncertain "
        "areas. Return a short prioritized checklist."
    ),
}

APP_CSS = """
:root { --ink: #e8eef7; --muted: #91a0b8; --panel: #121a27; --line: #273449; --cyan: #56d6e7; }
.gradio-container { max-width: 1560px !important; margin: 0 auto !important; background: #0a1019 !important; }
.app-shell { padding: 8px 4px 2px; }
.hero { border: 1px solid var(--line); border-radius: 18px; padding: 20px 24px; margin-bottom: 14px;
  background: radial-gradient(circle at 85% 10%, rgba(46,196,182,.16), transparent 34%),
              linear-gradient(135deg, #111b2a, #0d1521); }
.hero h1 { font-size: 28px; line-height: 1.15; margin: 0 0 7px; letter-spacing: -.02em; }
.hero p { color: var(--muted); margin: 0; }
.eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-size: 11px; font-weight: 700; }
.panel { border: 1px solid var(--line) !important; border-radius: 16px !important; padding: 14px !important;
  background: rgba(18,26,39,.88) !important; box-shadow: 0 14px 38px rgba(0,0,0,.18); }
.section-label h3 { margin: 0 0 8px !important; font-size: 15px !important; color: #cbd7e8; }
#observation-view { border-radius: 12px; overflow: hidden; }
#observation-view .image-container { min-height: 0 !important; aspect-ratio: 4 / 3; background: #05080d; }
#observation-view .image-container img { width: 100% !important; height: 100% !important; object-fit: cover !important; }
#status-line { min-height: 32px; border-left: 3px solid var(--cyan); padding: 6px 10px; color: var(--muted); }
.state-card { min-height: 148px; padding: 8px 12px; border-radius: 12px; background: #0c1420; }
.state-card h3 { font-size: 14px !important; color: var(--cyan); margin: 4px 0 9px !important; }
.nav-grid button { min-height: 42px; border-radius: 10px !important; font-weight: 650; }
.nav-primary { border-color: #2e7f89 !important; }
.manual-console { max-width: 560px; margin: 4px auto 0; }
.manual-console button { min-height: 46px; }
.key-hint { color: #91a0b8; text-align: center; font-size: 12px; margin: 8px 0 2px; }
.vlm-output { min-height: 285px; padding: 12px; border-radius: 12px; background: #09111c; border: 1px solid #223149; }
.vlm-status { color: var(--muted); min-height: 28px; }
.oracle-note { color: #8fa0ba; font-size: 12px; }
.footer-note { color: #73839d; text-align: center; font-size: 12px; margin: 10px 0 2px; }
"""

SCENES = (
    [f"FloorPlan{i}" for i in range(1, 31)]
    + [f"FloorPlan{i}" for i in range(201, 231)]
    + [f"FloorPlan{i}" for i in range(301, 331)]
    + [f"FloorPlan{i}" for i in range(401, 431)]
)
HISTORY_HEADERS = ["Step", "Action", "Success", "ms", "x", "z", "yaw"]

KEYBOARD_JS = """
() => {
  if (window.__vlmSimKeyboardInstalled) return;
  window.__vlmSimKeyboardInstalled = true;
  const keys = {
    w: 'move-ahead', s: 'move-back', a: 'move-left', d: 'move-right',
    q: 'rotate-left', e: 'rotate-right', arrowup: 'look-up', arrowdown: 'look-down',
    c: 'crouch', x: 'stand'
  };
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    const id = keys[event.key.toLowerCase()];
    if (!id) return;
    const button = document.querySelector(`#${id} button, button#${id}`);
    if (button && !button.disabled) {
      event.preventDefault();
      button.click();
    }
  });
}
"""


class BackendPool:
    """Lazily load and reuse one backend instance per backend/model pair."""

    def __init__(self):
        self._backends: dict[tuple[str, str], VLMBackend] = {}
        self._lock = threading.Lock()

    def get(
        self, name: str, model: str, metadata: dict[str, Any]
    ) -> tuple[VLMBackend, bool]:
        if name == "metadata":
            return make_backend(name, model, metadata), False
        key = (name, model)
        with self._lock:
            loaded_now = key not in self._backends
            if loaded_now:
                self._backends[key] = make_backend(name, model)
            return self._backends[key], loaded_now


def _agent_markdown(metadata: dict[str, Any]) -> str:
    agent = metadata.get("agent", {})
    position = agent.get("position", {})
    rotation = agent.get("rotation", {})
    return (
        "### Agent State\n"
        f"**Position**  `x {position.get('x', 0):.2f}` · "
        f"`y {position.get('y', 0):.2f}` · `z {position.get('z', 0):.2f}`  \n"
        f"**Yaw**  `{rotation.get('y', 0):.1f}°`  \n"
        f"**Horizon**  `{agent.get('cameraHorizon', 0):.1f}°`  \n"
        f"**Standing**  `{agent.get('isStanding', False)}`"
    )


def _action_markdown(metadata: dict[str, Any], elapsed_ms: float = 0.0) -> str:
    success = bool(metadata.get("lastActionSuccess", True))
    marker = "●" if success else "×"
    error = metadata.get("errorMessage") or "None"
    return (
        "### Last Transition\n"
        f"**Action**  `{metadata.get('lastAction', 'Reset')}`  \n"
        f"**Result**  {marker} `{'success' if success else 'failed'}`  \n"
        f"**Latency**  `{elapsed_ms:.1f} ms`  \n"
        f"**Error**  `{error}`"
    )


def _object_markdown(debugger: SimulatorDebugger, object_id: str | None) -> str:
    obj = next(
        (item for item in debugger.visible_objects() if item.get("objectId") == object_id), None
    )
    if obj is None:
        return "### Object Inspector\nSelect a visible object to inspect simulator ground truth."
    position = obj.get("position", {})
    return (
        "### Object Inspector\n"
        f"**{obj.get('objectType')}**  \n`{obj.get('objectId')}`  \n\n"
        f"**Distance** `{float(obj.get('distance', 0)):.2f} m`  \n"
        f"**Position** `({position.get('x', 0):.2f}, {position.get('y', 0):.2f}, "
        f"{position.get('z', 0):.2f})`  \n"
        f"**Pickupable** `{obj.get('pickupable', False)}` · "
        f"**Moveable** `{obj.get('moveable', False)}`  \n"
        f"**Openable** `{obj.get('openable', False)}` · "
        f"**Open** `{obj.get('isOpen', False)}`"
    )


def build_app(backend_name: str = "transformers", model: str | None = None):
    # Import after main() bootstraps the headless X display.
    import gradio as gr

    debugger = SimulatorDebugger()
    initial_backend = "transformers" if backend_name == "qwen" else backend_name
    if initial_backend not in BACKEND_OPTIONS:
        raise ValueError(f"Unsupported backend: {backend_name}")
    model_options = load_model_options()
    initial_model = model or model_options[initial_backend][0]
    backend_pool = BackendPool()

    def session_payload(selected_object: str | None = None, elapsed_ms: float = 0.0):
        if debugger.observation is None:
            raise gr.Error("Reset a scene first.")
        metadata = debugger.observation.metadata
        visible = debugger.visible_objects()
        object_ids = [str(obj.get("objectId")) for obj in visible]
        if selected_object not in object_ids:
            selected_object = object_ids[0] if object_ids else None
        step = max(0, len(debugger.history) - 1)
        status = (
            f"**Live session** · step `{step}` · `{metadata.get('sceneName', 'unknown')}` · "
            f"{len(visible)} visible objects · last action "
            f"`{metadata.get('lastAction', 'Reset')}`"
        )
        return (
            debugger.observation.image,
            status,
            _agent_markdown(metadata),
            _action_markdown(metadata, elapsed_ms),
            gr.Dropdown(choices=object_ids, value=selected_object),
            _object_markdown(debugger, selected_object),
            debugger.top_down_map(),
            debugger.history_rows(),
            debugger.history_gallery(),
            f"Observation changed to step `{step}` — run VLM analysis when ready.",
        )

    def reset_scene(scene: str):
        started = time.perf_counter()
        debugger.reset(scene)
        return session_payload(elapsed_ms=(time.perf_counter() - started) * 1000)

    def take_action(action: str, selected_object: str | None):
        if debugger.simulator is None:
            raise gr.Error("Reset a scene before taking an action.")
        started = time.perf_counter()
        debugger.step(action)
        return session_payload(selected_object, (time.perf_counter() - started) * 1000)

    def analyze(selected_backend: str, selected_model: str, prompt: str):
        if not prompt.strip():
            raise gr.Error("Enter a VLM prompt first.")
        selected_model = selected_model.strip()
        if not selected_model:
            raise gr.Error("Select or enter a model first.")
        try:
            image, metadata, step = debugger.analysis_snapshot()
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc
        started = time.perf_counter()
        try:
            instance, loaded_now = backend_pool.get(selected_backend, selected_model, metadata)
            answer = instance.describe(image, prompt.strip())
        except Exception as exc:  # keep the debugger usable if VLM initialization fails
            return (
                f"### VLM error\n`{type(exc).__name__}: {exc}`",
                f"Analysis failed for step `{step}`. Simulator controls remain available.",
            )
        elapsed = time.perf_counter() - started
        load_note = " · model loaded on this request" if loaded_now else ""
        return (
            answer,
            f"Analyzed step `{step}` with `{selected_backend}` / `{selected_model}` "
            f"in `{elapsed:.2f}s`{load_note}.",
        )

    def select_backend(name: str):
        choices = model_options[name]
        return gr.Dropdown(choices=choices, value=choices[0], allow_custom_value=True)

    def select_prompt(name: str):
        return PROMPTS.get(name, DEFAULT_PROMPT)

    theme = gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    )
    with gr.Blocks(title="Embodied VLM Studio", css=APP_CSS, theme=theme, js=KEYBOARD_JS) as app:
        gr.HTML(
            '<div class="app-shell"><div class="hero">'
            '<div class="eyebrow">AI2-THOR · Qwen3-VL · RTX 4090</div>'
            '<h1>Embodied VLM Studio</h1>'
            '<p>Drive the simulator, inspect oracle state, and ask a vision-language model '
            'to reason over the current egocentric observation.</p></div></div>'
        )
        with gr.Row(elem_classes="panel"):
            scene = gr.Dropdown(SCENES, value="FloorPlan1", label="Scene", scale=5)
            reset = gr.Button("Reset Session", variant="primary", scale=1)
        status = gr.Markdown("No active scene — reset a session to begin.", elem_id="status-line")

        with gr.Row(equal_height=False):
            with gr.Column(scale=8, elem_classes="panel"):
                gr.Markdown("### Egocentric Observation", elem_classes="section-label")
                observation = gr.Image(
                    label=None, type="pil", interactive=False, elem_id="observation-view"
                )
                with gr.Accordion("Action history and observation timeline", open=False):
                    history = gr.Dataframe(
                        headers=HISTORY_HEADERS,
                        datatype=["number", "str", "bool", "number", "number", "number", "number"],
                        interactive=False,
                        wrap=True,
                    )
                    thumbnails = gr.Gallery(label="Observation Timeline", columns=5, height=190)

            with gr.Column(scale=6, elem_classes="panel"):
                with gr.Tabs():
                    with gr.Tab("Manual Policy", id="manual"):
                        gr.Markdown(
                            "Directly control the embodied agent. Navigation updates the observation "
                            "without automatically running the VLM."
                        )
                        with gr.Column(elem_classes=["nav-grid", "manual-console"]):
                            with gr.Row():
                                move_ahead = gr.Button(
                                    "[Move Ahead] (W)",
                                    elem_classes="nav-primary",
                                    elem_id="move-ahead",
                                )
                                move_left = gr.Button("[Move Left] (A)", elem_id="move-left")
                                move_back = gr.Button("[Move Back] (S)", elem_id="move-back")
                                move_right = gr.Button("[Move Right] (D)", elem_id="move-right")
                            with gr.Row():
                                rotate_left = gr.Button("[Rotate Left] (Q)", elem_id="rotate-left")
                                rotate_right = gr.Button("[Rotate Right] (E)", elem_id="rotate-right")
                            with gr.Row():
                                look_up = gr.Button("[Look Up] (↑)", elem_id="look-up")
                                look_down = gr.Button("[Look Down] (↓)", elem_id="look-down")
                            with gr.Row():
                                crouch = gr.Button("[Crouch] (C)", elem_id="crouch")
                                stand = gr.Button("[Stand] (X)", elem_id="stand")
                        gr.HTML(
                            '<div class="key-hint">Keyboard: W/A/S/D move · Q/E rotate · '
                            '↑/↓ look · C/X posture</div>'
                        )
                    with gr.Tab("VLM Copilot", id="vlm"):
                        with gr.Row():
                            backend_selector = gr.Dropdown(
                                BACKEND_OPTIONS,
                                value=initial_backend,
                                label="Backend",
                                interactive=True,
                            )
                            model_selector = gr.Dropdown(
                                model_options[initial_backend],
                                value=initial_model,
                                label="Model",
                                allow_custom_value=True,
                                interactive=True,
                            )
                        gr.Markdown(
                            "The model sees RGB pixels only; oracle metadata is shown separately. "
                            "Model IDs may be selected from the list or entered directly."
                        )
                        preset = gr.Dropdown(
                            list(PROMPTS), value="Scene understanding", label="Analysis preset"
                        )
                        prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt", lines=6)
                        analyze_button = gr.Button("Analyze Current Observation", variant="primary")
                        vlm_status = gr.Markdown(
                            "Reset a scene, then analyze the current step.", elem_classes="vlm-status"
                        )
                        vlm_output = gr.Markdown(
                            "### Awaiting analysis\nThe result will appear here without blocking navigation history.",
                            elem_classes="vlm-output",
                        )

        with gr.Column(elem_classes="panel"):
            with gr.Tabs():
                with gr.Tab("Spatial Map", id="map"):
                    gr.Markdown(
                        "Reachable positions, visible object centers, agent heading, and trajectory.",
                        elem_classes="oracle-note",
                    )
                    top_down = gr.Image(
                        label=None, type="pil", interactive=False, height=430, elem_id="map-view"
                    )
                with gr.Tab("Oracle Inspector", id="oracle"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2):
                            gr.Markdown(
                                "Simulator ground truth for debugging—never passed to the visual model.",
                                elem_classes="oracle-note",
                            )
                            visible_objects = gr.Dropdown([], label="Visible object")
                        with gr.Column(scale=3):
                            object_inspector = gr.Markdown(
                                "### Object Inspector\nSelect a visible object after reset."
                            )
                with gr.Tab("Current Stage", id="state"):
                    with gr.Row():
                        agent_state = gr.Markdown("### Agent State\n—", elem_classes="state-card")
                        action_state = gr.Markdown(
                            "### Last Transition\n—", elem_classes="state-card"
                        )
        gr.HTML('<div class="footer-note">VLM output is model-generated. Verify actions against simulator state.</div>')

        outputs = [
            observation,
            status,
            agent_state,
            action_state,
            visible_objects,
            object_inspector,
            top_down,
            history,
            thumbnails,
            vlm_status,
        ]
        reset.click(
            reset_scene,
            scene,
            outputs,
            api_name="reset_scene",
            show_progress="hidden",
        )
        for button, action in (
            (move_ahead, "MoveAhead"),
            (move_back, "MoveBack"),
            (move_left, "MoveLeft"),
            (move_right, "MoveRight"),
            (rotate_left, "RotateLeft"),
            (rotate_right, "RotateRight"),
            (look_up, "LookUp"),
            (look_down, "LookDown"),
            (crouch, "Crouch"),
            (stand, "Stand"),
        ):
            button.click(
                lambda selected, a=action: take_action(a, selected),
                visible_objects,
                outputs,
                api_name=f"action_{action}",
                show_progress="hidden",
            )
        visible_objects.change(
            lambda object_id: _object_markdown(debugger, object_id),
            visible_objects,
            object_inspector,
        )
        preset.change(select_prompt, preset, prompt)
        backend_selector.change(select_backend, backend_selector, model_selector)
        analyze_button.click(
            analyze,
            [backend_selector, model_selector, prompt],
            [vlm_output, vlm_status],
            api_name="analyze_vlm",
            show_progress="hidden",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AI2-THOR embodied VLM studio")
    parser.add_argument(
        "--backend",
        choices=["transformers", "qwen", "openai", "metadata"],
        default="transformers",
    )
    parser.add_argument("--model", help="Initial model; may also be selected in the UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if os.environ.get("AI2THOR_PLATFORM", "Linux64") == "Linux64" and not os.environ.get("DISPLAY"):
        os.execvp(
            "xvfb-run",
            [
                "xvfb-run",
                "-a",
                "-s",
                "-screen 0 1280x720x24",
                sys.executable,
                "-m",
                "vlm_sim.web",
                *sys.argv[1:],
            ],
        )
    build_app(args.backend, args.model).queue(default_concurrency_limit=2).launch(
        server_name=args.host, server_port=args.port
    )


if __name__ == "__main__":
    main()
