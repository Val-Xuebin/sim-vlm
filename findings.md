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
