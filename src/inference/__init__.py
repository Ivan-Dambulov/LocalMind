"""Inference package — import submodules directly to avoid hard dependency at import time."""

from src.inference.base import BaseLLM
from src.inference.hardware import (
    detect_hardware,
    recommend_gpu_layers,
    recommend_context_size,
    format_hardware_summary,
    HardwareInfo,
    GPUInfo,
)

__all__ = [
    "BaseLLM",
    "detect_hardware",
    "recommend_gpu_layers",
    "recommend_context_size",
    "format_hardware_summary",
    "HardwareInfo",
    "GPUInfo",
]
