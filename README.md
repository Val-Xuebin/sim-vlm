# AI2-THOR + VLM Demo

A compact AI2-THOR debugger with selectable local and remote vision-language models. The UI provides live navigation, RGB observations, simulator metadata, a spatial map, and on-demand VLM analysis.

## Requirements

- Linux with Python 3.10 or 3.11
- NVIDIA GPU and a recent driver (tested on RTX 4090, driver 580.105.08)
- At least 15 GiB of free disk space for the GPU environment and 2B model

## Install

```bash
git clone https://github.com/Val-Xuebin/sim-vlm.git
cd sim-vlm
INSTALL_LOCAL_VLM=1 bash scripts/create_env.sh
bash scripts/download_model.sh
```

The setup creates `.venv`, installs PyTorch with CUDA 12.8, and stores the model under `.cache/huggingface`. A newer NVIDIA driver can run the bundled CUDA 12.8 runtime.

## Verify

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
pytest -q
bash scripts/run_demo.sh --backend metadata --run-id smoke
bash scripts/run_demo.sh --backend qwen --run-id qwen-smoke
```

Run artifacts are written to `outputs/<run-id>/`.

## Start the UI

```bash
.venv/bin/vlm-sim-ui --backend transformers --host 127.0.0.1 --port 7860
```

From your Mac, forward the server port:

```bash
ssh -L 7860:localhost:7860 <user>@<server>
```

Open `http://localhost:7860`. Select **Backend** and **Model** in VLM Copilot. Models load only when **Analyze Current Observation** is clicked and are reused by backend/model pair.

## Autonomous VLM Policy

Reset a scene, open **VLM Copilot → Autonomous VLM Policy**, enter a task, enable VLM Control, and click **Start VLM Control**.

At each policy step the VLM receives:

- the current RGB observation;
- the task and confidence threshold;
- a compact memory of prior observations, new information, actions, and confidence values.

The VLM returns a strict JSON decision. The simulator executes its action and supplies the new observation on the next step. The run stops when the confidence threshold is reached, the model selects `Stop`, reports `blocked`, reaches the maximum step count, or the user clicks **Stop**. The final RGB observation, raw VLM output, and complete policy trace remain visible.

Manual actions and scene reset are rejected while VLM Control is active. Keep a finite maximum step count: model actions and confidence are not guaranteed to be correct.

## Model backends

- `metadata`: simulator-only smoke test; no pixel inference or GPU required.
- `transformers`: local Hugging Face VLM using `AutoProcessor` and `AutoModelForImageTextToText`; CUDA is required.
- `openai`: any vision model exposed through an OpenAI-compatible API; no local GPU required.

The Model dropdown accepts custom IDs. Persistent presets live in [`configs/models.json`](configs/models.json):

```json
{
  "metadata": ["simulator-metadata"],
  "transformers": ["Qwen/Qwen3-VL-2B-Instruct", "organization/model-id"],
  "openai": ["gpt-4.1-mini", "provider/model-name"]
}
```

Use another configuration file with:

```bash
VLM_SIM_MODEL_CONFIG=/path/to/models.json .venv/bin/vlm-sim-ui
```

For an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-endpoint/v1
.venv/bin/vlm-sim-ui --backend openai
```

Local models must support the Transformers image-text auto classes and chat template. Models requiring custom preprocessing should be added as a new `VLMBackend` adapter in `src/vlm_sim/backends.py`.

## CLI

```bash
bash scripts/run_demo.sh --backend transformers \
  --action RotateRight --action MoveAhead --run-id moved-view
```

`qwen` remains a CLI alias for `transformers` for backward compatibility.
