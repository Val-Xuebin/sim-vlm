# AI2-THOR + Qwen3-VL Demo

A compact simulator debugger for AI2-THOR with local Qwen3-VL scene analysis. The UI provides live navigation, RGB observations, simulator metadata, a spatial map, and on-demand VLM analysis.

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
.venv/bin/vlm-sim-ui --backend qwen --host 127.0.0.1 --port 7860
```

From your Mac, forward the server port:

```bash
ssh -L 7860:localhost:7860 <user>@<server>
```

Open `http://localhost:7860`. The model loads on the first **Analyze Current Observation** request and is reused afterward.

## CLI

```bash
bash scripts/run_demo.sh --backend qwen \
  --action RotateRight --action MoveAhead --run-id moved-view
```

The `metadata` backend is only a pipeline smoke test and does not inspect pixels. Use `qwen` for real visual inference.
