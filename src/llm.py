"""
LLM interface for LocalMind.
Selects CUDA / Metal / ROCm / CPU via settings + hardware + llama-cpp capability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Generator, List, Dict, Any

from src.config import settings
from src.inference.hardware import (
    detect_hardware,
    recommend_context_size,
    format_hardware_summary,
    HardwareInfo,
)
from src.inference.backends import probe_backends, resolve_gpu_layers
from src.inference.llama_runtime import LlamaRuntime


class LLM:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.llm_model_path
        self._runtime: Optional[LlamaRuntime] = None
        self.hardware: Optional[HardwareInfo] = None
        self.gpu_layers_used: int = 0
        self.context_size_used: int = 0
        self.backend: str = "none"
        self.warnings: list[str] = []

        if not self.model_path or not Path(self.model_path).exists():
            print("[llm] No valid GGUF model path – mock mode")
            return

        try:
            self.hardware = detect_hardware()
            print("[llm] Hardware detected:")
            print(format_hardware_summary(self.hardware))

            preferred = getattr(settings, "llm_preferred_backend", "auto") or "auto"
            configured_layers = int(getattr(settings, "llm_gpu_layers", -1))

            gpu_layers, backend, warnings = resolve_gpu_layers(
                preferred_backend=preferred,
                hardware=self.hardware,
                configured_layers=configured_layers,
            )
            self.warnings = warnings
            for w in warnings:
                print(f"[llm] Warning: {w}")

            context_size = min(
                recommend_context_size(self.hardware),
                settings.llm_context_size,
            )

            self._runtime = LlamaRuntime(
                model_path=self.model_path,
                context_size=context_size,
                gpu_layers=gpu_layers,
                temperature=settings.llm_temperature,
            )
            self.gpu_layers_used = gpu_layers
            self.context_size_used = context_size
            self.backend = backend if gpu_layers != 0 else "cpu"

            print(
                f"[llm] Ready | backend={self.backend} | "
                f"layers={gpu_layers} | context={context_size}"
            )
        except Exception as e:
            print(f"[llm] Failed to load model: {e}")
            self.warnings.append(str(e))

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        if self._runtime is None:
            return (
                "[MOCK LLM] Configure a valid GGUF model path "
                "(Settings tab or LLM_MODEL_PATH in .env)."
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
        messages = [
            {"role": "system", "content": "You are a careful and accurate assistant."},
            {"role": "user", "content": prompt},
        ]
        yield from self._runtime.stream(messages)

    @property
    def is_ready(self) -> bool:
        return self._runtime is not None

    def acceleration_info(self) -> dict[str, Any]:
        hw = self.hardware
        status = None
        try:
            status = probe_backends(hw)
        except Exception:
            pass

        info: dict[str, Any] = {
            "backend_in_use": self.backend,
            "gpu_layers_in_use": self.gpu_layers_used,
            "context_size_in_use": self.context_size_used,
            "model_ready": self.is_ready,
            "warnings": list(self.warnings),
            "preferred_backend": getattr(settings, "llm_preferred_backend", "auto"),
        }
        if status:
            info.update(
                {
                    "hardware_backend": status.hardware_backend,
                    "llama_gpu_offload": status.llama_gpu_offload,
                    "available_backends": status.available,
                    "recommended_backend": status.recommended,
                    "notes": status.notes,
                    "install_hints": status.install_hints,
                }
            )
        if hw:
            info["hardware_summary"] = format_hardware_summary(hw)
            info["gpus"] = [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "backend": g.backend,
                    "vram_mb": g.total_vram_mb,
                }
                for g in hw.gpus
            ]
        return info
