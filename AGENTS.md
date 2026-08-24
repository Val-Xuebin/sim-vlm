# Project Guide

- Keep source in `src/vlm_sim/`, reproducible setup in `scripts/`, and configs in `configs/`.
- Record every demo under `outputs/<run-id>/`; do not claim VLM results unless `response.md` exists.
- Default local model is `Qwen/Qwen3-VL-2B-Instruct`; keep model caches outside this NFS project when space permits.
- Default to AI2-THOR `Linux64` under Xvfb because the current NVIDIA Vulkan ICD is incompatible; retain `AI2THOR_PLATFORM=CloudRendering` as an opt-in path.
- Never commit API keys or large model weights.
- Before every final user reply, send a concise completion notification with `python3 ../bark/notify.py TITLE SUMMARY`; never include secrets or raw logs.
