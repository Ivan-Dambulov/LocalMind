"""
LLM interface for LocalMind.

Uses the inference code adapted from LocalBot (now living in src/inference/).
Provides a simple generate(prompt: str) -> str API that the agent expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Generator, List, Dict

from src.config import settings
from src.inference.hardware import (
    detect_hardware,
    recommend_gpu_layers,
    recommend_context_size,
    format_hardware_summary,
)
from src.inference.llama_runtime import LlamaRuntime


class LLM:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.llm_model_path
        self._runtime: Optional[LlamaRuntime] = None

        if not self.model_path or not Path(self.model_path).exists():
            print("[llm] No valid GGUF model path – running in mock mode")
            print("[llm] Set LLM_MODEL_PATH in .env or settings.llm_model_path")
            return

        try:
            hw = detect_hardware()
            print("[llm] Hardware detected:")
            print(format_hardware_summary(hw))

            gpu_layers = recommend_gpu_layers(hw)
            context_size = min(
                recommend_context_size(hw),
                settings.llm_context_size,
            )

            self._runtime = LlamaRuntime(
                model_path=self.model_path,
                context_size=context_size,
                gpu_layers=gpu_layers,
                temperature=settings.llm_temperature,
            )
            print(
                f"[llm] Ready | GPU layers={gpu_layers} | context={context_size}"
            )
        except Exception as e:
            print(f"[llm] Failed to load model: {e}")
            print("[llm] Falling back to mock mode")

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        if self._runtime is None:
            return (
                "[MOCK LLM] Configure a valid GGUF model path "
                "(settings.llm_model_path or LLM_MODEL_PATH in .env)."
            )

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful and accurate assistant. "
                    "Answer based only on the provided context when available. "
                    "Cite sources when you use them. "
                    "If the context is insufficient, say so clearly."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._runtime.generate(messages)

    def stream(self, prompt: str) -> Generator[str, None, None]:
        if self._runtime is None:
            yield "[MOCK LLM] No model loaded"
            return

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a careful and accurate assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        yield from self._runtime.stream(messages)

    @property
    def is_ready(self) -> bool:
        return self._runtime is not None
