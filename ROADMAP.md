# Roadmap

## Benchmark Tasks

- Evaluate VSI-Super task types in the simulator/VLM pipeline.
- Evaluate ESI benchmark task types with the same observation, action, and logging interface.
- Map each benchmark task to required observations, valid action space, success conditions, and metrics.
- Add reproducible evaluation configs, seeds, per-episode traces, aggregate metrics, and failure analysis.
- Compare single-observation inference with autonomous multi-observation policy runs.

## Streaming and Codec VLMs

- Evaluate Joy-VL and related streaming/codec vision-language models.
- Measure first-token latency, frame throughput, temporal context length, GPU memory, and task success.
- Compare raw-frame, sampled-frame, and codec/token-stream inputs under matched simulator trajectories.
- Add a streaming backend adapter without coupling model-specific preprocessing to the simulator core.
- Verify the exact Joy-VL repository/model identifier and runtime requirements before implementation.

## Evaluation Order

1. Define a shared episode and metric schema.
2. Implement benchmark adapters for VSI-Super and ESI.
3. Establish the current Qwen3-VL baseline.
4. Add the Joy-VL streaming/codec adapter.
5. Run matched evaluations and document limitations without inventing results.
