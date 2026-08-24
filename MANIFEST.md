# Manifest

- `src/vlm_sim/simulator.py`: AI2-THOR lifecycle, observation, metadata export.
- `src/vlm_sim/debugger.py`: persistent simulator session, navigation history, oracle inspection, map rendering, and immutable VLM snapshots.
- `src/vlm_sim/backends.py`: local Transformers, OpenAI-compatible, and metadata backends.
- `src/vlm_sim/autonomy.py`: structured VLM policy prompt, decision parser, memory, and stop rules.
- `src/vlm_sim/cli.py`: reproducible CLI demo and run manifest.
- `src/vlm_sim/web.py`: debugger UI with runtime backend/model selection and lazy instance reuse.
- `configs/models.json`: editable model presets shown in the UI.
- `tests/test_autonomy.py`: policy parsing, prompt-memory, action validation, and stop tests.
- `scripts/create_env.sh`: project-local venv setup with optional CUDA VLM dependencies.
- `scripts/download_model.sh`: resumable Qwen3-VL download using the working mirror endpoint.
- `scripts/run_demo.sh`: cache-aware demo launcher.
- `outputs/`: generated frames, metadata, model responses, and run manifests.
