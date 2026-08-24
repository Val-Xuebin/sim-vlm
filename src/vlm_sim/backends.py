from __future__ import annotations

import base64
import io
import json
import os
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class VLMBackend(ABC):
    @abstractmethod
    def describe(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class MetadataBackend(VLMBackend):
    """Dependency-light plumbing check; it deliberately does not inspect pixels."""

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = metadata or {}

    def describe(self, image: Image.Image, prompt: str) -> str:
        visible = self.metadata.get("visibleObjects", [])
        types = sorted({item.get("objectType", "Unknown") for item in visible})
        return json.dumps(
            {
                "summary": f"Simulator frame {image.width}x{image.height}",
                "objects": types,
                "hazards": [],
                "next_action": "RotateRight",
                "note": "metadata smoke-test backend; no pixel inference",
            },
            ensure_ascii=False,
        )


class QwenBackend(VLMBackend):
    def __init__(self, model: str, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen backend requires CUDA, but torch.cuda.is_available() is false")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.max_new_tokens = max_new_tokens

    def describe(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class OpenAICompatibleBackend(VLMBackend):
    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        self.model = model

    def describe(self, image: Image.Image, prompt: str) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def make_backend(name: str, model: str, metadata: dict[str, Any] | None = None) -> VLMBackend:
    if name == "qwen":
        return QwenBackend(model)
    if name == "openai":
        return OpenAICompatibleBackend(model)
    if name == "metadata":
        return MetadataBackend(metadata)
    raise ValueError(f"Unknown backend: {name}")
