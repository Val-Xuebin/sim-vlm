# AI2-THOR × VLM Demo

一个尽量小但可扩展的闭环：AI2-THOR 渲染第一视角 RGB，VLM 理解画面并输出场景摘要、物体、风险和下一步动作；同时保存 simulator metadata，方便对照 VLM 是否幻觉。

本机默认用 `Linux64 + Xvfb` 做无显示器渲染。若 NVIDIA Vulkan ICD 已修复，可设置 `AI2THOR_PLATFORM=CloudRendering`。

## 1. 创建环境

```bash
cd /root/shared-nvme/sim-vlm/workplace/vlm-sim-demo
bash scripts/create_env.sh
source .venv/bin/activate
```

默认只安装 simulator/UI 依赖，便于先验证链路。安装本地 GPU VLM 依赖时运行：

```bash
INSTALL_LOCAL_VLM=1 bash scripts/create_env.sh
bash scripts/download_model.sh
```

环境使用项目内 `.venv`、PyTorch CUDA 12.8 wheel、AI2-THOR 4.3.0、Transformers 4.57+ 和 Qwen3-VL。CUDA wheel 版本不需要与驱动显示的 CUDA 13.0 完全一致；新驱动可以运行较旧的 CUDA runtime。默认模型为适合快速演示的 `Qwen/Qwen3-VL-2B-Instruct`（BF16 权重约 4.27GB），可通过 `--model Qwen/Qwen3-VL-4B-Instruct` 切换 4B。

`download_model.sh` 默认使用可续传的 Hugging Face 镜像，并清除本机已失效的 localhost 代理变量。模型缓存位于 `.cache/huggingface/`。

## 2. 两级验证

先验证 simulator、渲染和文件链路（不加载模型）：

```bash
bash scripts/run_demo.sh --backend metadata --run-id smoke
```

再运行 4090 本地 VLM（首次会下载约 8 GB 权重）：

```bash
bash scripts/run_demo.sh --backend qwen --run-id qwen-floorplan1
```

结果位于 `outputs/<run-id>/`：`frame.png`、`metadata.json`、`response.md`、`run.json`。

可添加动作后再观察：

```bash
bash scripts/run_demo.sh --backend qwen \
  --action RotateRight --action MoveAhead --run-id moved-view
```

## 3. 展示 UI

```bash
.venv/bin/vlm-sim-ui \
  --backend qwen --host 127.0.0.1 --port 7860
```

然后从本地做 SSH 端口转发：`ssh -L 7860:localhost:7860 <server>`，浏览器访问 `http://localhost:7860`。

UI 采用持久 simulator session：左侧显示第一视角画面、agent 状态和手动导航；右侧提供 VLM Copilot、Oracle Inspector 和 Spatial Map。Qwen 模型在首次点击 **Analyze Current Observation** 时懒加载，后续分析复用同一模型；导航动作不会自动触发 VLM，因此不会阻塞 simulator 调试。

VLM 只接收当前 RGB 像素。AI2-THOR 的可见物体、位置和可达区域仅显示在 Oracle/Map 标签页，不会注入视觉模型 prompt，便于检查幻觉和感知差异。

## 4. OpenAI-compatible 后端

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-endpoint/v1  # 官方 OpenAI 可不设置
bash scripts/run_demo.sh --backend openai --model <vision-model-name>
```

`metadata` 后端只用于管线 smoke test，不读取像素，输出不会被当作 VLM 实验结果。
