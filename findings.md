# Findings

## 2026-08-24 initial machine check

- Host GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 580.105.08, compute capability 8.9.
- The default Codex sandbox did not expose a usable NVIDIA driver, while the host execution context did.
- No X display is configured, so AI2-THOR is configured with `CloudRendering`.
- Project NFS had about 7.2GB free and the container overlay only about 0.5GB free. A full CUDA environment plus Qwen3-VL weights needs additional storage; the core `.venv` and simulator smoke test are installed separately from optional local-VLM dependencies.
- Storage was expanded to about 27GB free. AI2-THOR CloudRendering could only enumerate Mesa llvmpipe because the mounted NVIDIA Vulkan ICD failed `vkCreateInstance`; use Linux64 under Xvfb for simulator rendering while CUDA remains available for Qwen3-VL.

## 2026-08-24 verified demo

- Command: `bash scripts/run_demo.sh --backend metadata --run-id smoke-linux64`.
- Simulator result: AI2-THOR `FloorPlan1` rendered a valid 640x480 kitchen RGB frame under Linux64 + Xvfb; artifacts are in `outputs/smoke-linux64/`.
- Local VLM environment: PyTorch 2.11.0+cu128, Transformers 4.57.6, CUDA available on RTX 4090.
- Model: `Qwen/Qwen3-VL-2B-Instruct`, BF16, about 4.0GB cached under `.cache/huggingface/`.
- Isolated model load took 10.73 seconds and peak allocated CUDA memory was about 3.99GiB.
- Existing-frame inference took 17.763 seconds and correctly identified the kitchen, refrigerator, cabinets, countertop, and toaster oven.
- End-to-end command: `bash scripts/run_demo.sh --backend qwen --run-id qwen3-floorplan1`.
- End-to-end wall time was 46.846 seconds including simulator start, model load, inference, and artifact writing. Result is in `outputs/qwen3-floorplan1/`.

## 2026-08-24 headless web UI validation

- The observation originally did not appear because rendering and VLM inference shared one Gradio callback. A Qwen/Torch failure discarded the already-rendered image along with the whole callback response.
- The UI now separates `Render Scene` from optional `Analyze Observation`; rendering has no Torch or VLM dependency.
- Validation command: `.venv/bin/vlm-sim-ui --backend metadata --host 127.0.0.1 --port 17860`, followed by a `gradio_client` call to `/render_scene` with `FloorPlan1`.
- Result: RGB image delivered through Gradio at 640x480, scene `FloorPlan1_physics`, four visible objects. The temporary validation server was stopped after the check.

## 2026-08-24 simulator debugger draft archived

- A debugger prototype with persistent navigation, extra sensor views, object inspection, top-down reachability, and history was implemented.
- Under `Linux64 + Xvfb`, Unity reported Mesa llvmpipe and about 0.03 FPS with extra depth/instance render passes. Reset did not finish within several minutes at either 640x480 or 320x240.
- The extra-sensor draft and failure note are preserved at `../../archived/vlm-sim-debugger-draft-20260824/`.
- The active UI retains the debugger with RGB observation, navigation, agent/action state, visible-object inspector, reachable-position map, trajectory, and observation history. End-to-end validation reset `FloorPlan1` in about 9 seconds and successfully applied `RotateRight`; tests produced `1 passed`.

## 2026-08-24 debugger UI and VLM integration

- Reorganized the active interface into an observation/control workspace plus VLM Copilot, Oracle Inspector, and Spatial Map tabs; history is collapsed by default.
- Removed hidden sensor/overlay states from the active interaction path and preserved the stable RGB-only Linux64 + Xvfb renderer.
- VLM inference is explicit and lazy-loaded. Each result is tied to the simulator step from which its immutable RGB snapshot was captured; actions mark the previous result stale without blocking navigation.
- Direct UI launch now defaults `HF_HOME` to the project cache and enables offline loading when the Qwen3-VL cache exists. This prevents accidental downloads to the nearly-full container root filesystem.
- Metadata-backend regression: `/reset_scene` returned all 10 debugger outputs and `/analyze_vlm` returned a step-linked response.
- Qwen-backend regression on `FloorPlan1`: reset took 8.27 seconds; first VLM analysis including model load took 15.00 seconds. Qwen correctly described the refrigerator, cabinets, countertop, appliance, and tiled backsplash.
- UI HTTP and Gradio API tests used temporary localhost ports and the servers were stopped after validation.

## 2026-08-25 configurable VLM backends

- Added runtime Backend and Model selectors. Model presets are read from `configs/models.json`; custom model IDs remain accepted directly in the UI.
- Backend instances are lazy-loaded and cached by `(backend, model)`. Switching models does not restart the simulator, and returning to an already loaded pair reuses it.
- `transformers` supports compatible Hugging Face image-text auto models; `qwen` remains a CLI alias. `openai` supports remote OpenAI-compatible vision endpoints, while `metadata` remains the no-GPU smoke test.
- Regression command: `HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/vlm-sim ask outputs/smoke-linux64/frame.png --backend transformers --model Qwen/Qwen3-VL-2B-Instruct --prompt "Name the room type in one short phrase."`.
- Result: `kitchen`. UI configuration, CLI alias, callback registration, Python compilation, and pytest (`1 passed`) also succeeded.

## 2026-08-25 autonomous VLM policy

- Archived stable commit `e057197` as Git tag `pre-autonomous-vlm-20260825` and workspace snapshot `archived/vlm-sim-demo-pre-autonomous-20260825/` before implementation.
- Added an opt-in task policy that loops over current RGB observation → structured VLM JSON → validated AI2-THOR action → new observation. Up to eight compressed prior decisions are included in each prompt.
- Stop conditions: configured confidence threshold, `Stop`, `blocked`, user cancellation, policy error, or a hard maximum of 50 steps. Scene reset and manual actions are rejected while the loop is active.
- Metadata end-to-end API validation reset `FloorPlan1`, returned final observation and trace, and stopped safely on its non-policy response.
- Qwen3-VL-2B end-to-end API validation task: `Determine whether a refrigerator is visible.`, threshold `0.75`, maximum `2` steps. First inference including model load took 14.28 seconds; output reported the refrigerator visible, confidence `0.95`, status `completed`, action `Stop`, and the loop stopped at the threshold without moving.

## 2026-08-25 task-level interaction actions

- Tagged stable pre-change commit as `pre-interaction-actions-20260825`.
- Expanded the shared Manual/VLM action catalog to 29 simulator actions plus policy `Stop`. Target actions resolve a VLM-provided visual object type or Manual object ID against visible metadata, then enforce object capability, state, held-object preconditions, and bounded parameters.
- Manual controls are grouped by action class and dynamically enabled for the selected target. Keyboard shortcuts use unmodified and Shift-modified keys and remain disabled while typing into form controls.
- Linux64 + Xvfb API validation on `FloorPlan1`: selected a visible Cabinet, executed `OpenObject` successfully in 378.1 ms, then `CloseObject` successfully in 371.7 ms. The temporary validation server was stopped afterward.
- Qwen grounding trial requested `Open a visible cabinet and confirm that it is open.` In the first run, Qwen emitted `OpenObject`, visual target `cabinet`, and `openness=0.5`; grounding resolved the real Cabinet ID and AI2-THOR executed it successfully. A repeated Open was rejected because metadata already reported the cabinet open, demonstrating state gating.
- A second stochastic trial did not complete the task: Qwen selected `LookDown` three times and hit the camera-horizon limit. The controller bounded the run at three steps and recorded the final simulator failure. The policy prompt now explicitly maps missing upper/lower/side regions to camera motions and discourages repeated view actions without new evidence; model planning is still not guaranteed.
