# Manifest

- `src/vlm_sim/simulator.py`: AI2-THOR lifecycle, observation, metadata export.
- `src/vlm_sim/debugger.py`: persistent simulator session, navigation history, oracle inspection, map rendering, and immutable VLM snapshots.
- `src/vlm_sim/backends.py`: local Qwen3-VL, OpenAI-compatible, and metadata smoke-test backends.
- `src/vlm_sim/cli.py`: reproducible CLI demo and run manifest.
- `src/vlm_sim/web.py`: reorganized debugger UI with a lazy-loaded VLM Copilot.
- `scripts/create_env.sh`: project-local venv setup with optional CUDA VLM dependencies.
- `scripts/download_model.sh`: resumable Qwen3-VL download using the working mirror endpoint.
- `scripts/run_demo.sh`: cache-aware demo launcher.
- `outputs/`: generated frames, metadata, model responses, and run manifests.
