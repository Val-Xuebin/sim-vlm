from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .backends import make_backend
from .simulator import ThorSimulator, compact_metadata, save_observation

DEFAULT_PROMPT = (
    "Describe the room, list important visible objects, identify possible hazards, and recommend "
    "one useful next robot action. Return concise JSON with keys summary, objects, hazards, next_action."
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_demo(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    started = time.time()
    with ThorSimulator(args.scene, args.width, args.height) as simulator:
        observation = simulator.observe()
        if args.action:
            for action in args.action:
                observation = simulator.step(action)
        image_path, metadata_path = save_observation(observation, output_dir)

    compact = compact_metadata(observation.metadata)
    backend = make_backend(args.backend, args.model, compact)
    answer = backend.describe(observation.image, args.prompt)
    response_path = output_dir / "response.md"
    response_path.write_text(answer + "\n", encoding="utf-8")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "actions": args.action or [],
        "backend": args.backend,
        "model": args.model,
        "prompt": args.prompt,
        "seed": None,
        "image": str(image_path),
        "metadata": str(metadata_path),
        "response": str(response_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
    }
    (output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(answer)
    print(f"\nArtifacts: {output_dir.resolve()}")
    return output_dir


def ask_image(args: argparse.Namespace) -> None:
    image = Image.open(args.image).convert("RGB")
    backend = make_backend(args.backend, args.model)
    print(backend.describe(image, args.prompt))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI2-THOR + VLM scene understanding")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="render a THOR scene and ask a VLM")
    demo.add_argument("--scene", default="FloorPlan1")
    demo.add_argument("--width", type=int, default=640)
    demo.add_argument("--height", type=int, default=480)
    demo.add_argument("--action", action="append", help="AI2-THOR action; repeatable")
    demo.add_argument(
        "--backend",
        choices=["transformers", "qwen", "openai", "metadata"],
        default="transformers",
    )
    demo.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    demo.add_argument("--prompt", default=DEFAULT_PROMPT)
    demo.add_argument("--output-dir", default="outputs")
    demo.add_argument("--run-id")
    demo.set_defaults(func=run_demo)

    ask = sub.add_parser("ask", help="ask a VLM about an existing image")
    ask.add_argument("image")
    ask.add_argument(
        "--backend", choices=["transformers", "qwen", "openai"], default="transformers"
    )
    ask.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ask.add_argument("--prompt", default=DEFAULT_PROMPT)
    ask.set_defaults(func=ask_image)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
