"""
Detect which LLM acceleration backends are usable.

Two layers:
1. Hardware present (NVIDIA / AMD / Apple / Intel)
2. llama-cpp-python build supports GPU offload

The installed wheel determines CUDA vs Metal vs ROCm — we cannot switch
backends at runtime without reinstalling llama-cpp-python. We *can*:
- choose CPU vs GPU offload
- recommend the right backend
- warn when the user picks a backend their install does not support
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.inference.hardware import detect_hardware, HardwareInfo, recommend_gpu_layers


BACKEND_CHOICES = ("auto", "cuda", "metal", "rocm", "cpu")


@dataclass
class BackendStatus:
    hardware_backend: str  # from GPU detection: cuda|metal|rocm|cpu
    llama_gpu_offload: bool
    available: List[str] = field(default_factory=list)  # selectable backends that make sense
    recommended: str = "cpu"
    notes: List[str] = field(default_factory=list)
    install_hints: dict = field(default_factory=dict)


def _llama_gpu_offload_supported() -> bool:
    try:
        from llama_cpp import llama_supports_gpu_offload
        return bool(llama_supports_gpu_offload())
    except Exception:
        try:
            # Older builds: try a probe
            import llama_cpp  # noqa: F401
            return False
        except Exception:
            return False


def probe_backends(hardware: Optional[HardwareInfo] = None) -> BackendStatus:
    hw = hardware or detect_hardware()
    hw_backend = hw.recommended_backend or "cpu"
    llama_gpu = _llama_gpu_offload_supported()

    available = ["cpu"]
    notes: list[str] = []
    hints = {
        "cuda": (
            "Windows/Linux NVIDIA:\n"
            'pip uninstall llama-cpp-python -y\n'
            'set CMAKE_ARGS=-DGGML_CUDA=on\n'
            "pip install llama-cpp-python --force-reinstall --no-cache-dir"
        ),
        "metal": (
            "macOS Apple Silicon:\n"
            "pip uninstall llama-cpp-python -y\n"
            'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
        ),
        "rocm": (
            "Linux AMD ROCm:\n"
            "pip uninstall llama-cpp-python -y\n"
            'CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
        ),
        "cpu": "Default pip install llama-cpp-python (CPU).",
    }

    # Hardware-backed options
    vendors = {g.vendor for g in hw.gpus}
    backends_hw = {g.backend for g in hw.gpus if g.backend}

    if "cuda" in backends_hw or "nvidia" in vendors:
        available.append("cuda")
        if not llama_gpu:
            notes.append(
                "NVIDIA GPU detected, but this llama-cpp-python build may lack CUDA. "
                "Reinstall with GGML_CUDA=on to use it."
            )
    if "metal" in backends_hw or hw.is_apple_silicon:
        available.append("metal")
        if not llama_gpu and hw.is_apple_silicon:
            notes.append(
                "Apple Silicon detected. Prefer a Metal-enabled llama-cpp-python build."
            )
    if "rocm" in backends_hw or "amd" in vendors:
        available.append("rocm")
        if not llama_gpu:
            notes.append(
                "AMD GPU detected, but ROCm support depends on a HIP-enabled llama-cpp build."
            )

    # Intel: detect only as note (llama-cpp SYCL is uncommon in default wheels)
    if any("intel" in (g.name or "").lower() for g in hw.gpus):
        notes.append(
            "Intel GPU detected. Official llama-cpp-python wheels rarely include SYCL; "
            "CPU or a custom SYCL build is typical."
        )

    # Recommended
    if hw_backend in ("cuda", "metal", "rocm") and hw_backend in available:
        recommended = hw_backend
    else:
        recommended = "cpu"

    # Deduplicate available preserving order
    seen = set()
    avail_unique = []
    for b in available:
        if b not in seen:
            seen.add(b)
            avail_unique.append(b)

    return BackendStatus(
        hardware_backend=hw_backend,
        llama_gpu_offload=llama_gpu,
        available=avail_unique,
        recommended=recommended,
        notes=notes,
        install_hints=hints,
    )


def resolve_gpu_layers(
    preferred_backend: str,
    hardware: Optional[HardwareInfo] = None,
    configured_layers: int = -1,
) -> tuple[int, str, list[str]]:
    """
    Returns (n_gpu_layers, backend_label, warnings).

    preferred_backend: auto|cuda|metal|rocm|cpu
    configured_layers: -1 auto layers, 0 force CPU layers, >0 fixed count
    """
    hw = hardware or detect_hardware()
    status = probe_backends(hw)
    preferred = (preferred_backend or "auto").lower().strip()
    warnings: list[str] = []

    if preferred == "auto":
        preferred = status.recommended

    if preferred not in BACKEND_CHOICES:
        warnings.append(f"Unknown backend '{preferred}', falling back to auto.")
        preferred = status.recommended

    # Force CPU
    if preferred == "cpu" or configured_layers == 0:
        return 0, "cpu", warnings

    # User asked for a GPU backend that hardware doesn't have
    if preferred in ("cuda", "metal", "rocm") and preferred not in status.available:
        warnings.append(
            f"Backend '{preferred}' is not available on this machine. Using CPU."
        )
        return 0, "cpu", warnings

    # Hardware OK but llama-cpp may not support offload
    if not status.llama_gpu_offload and preferred != "cpu":
        warnings.append(
            f"Preferred backend '{preferred}' needs a GPU-enabled llama-cpp-python build. "
            f"Currently GPU offload reports unsupported — using CPU until you reinstall."
        )
        # Still try -1 in case the probe is wrong (Metal sometimes works anyway)
        if preferred == "metal" and hw.is_apple_silicon:
            layers = recommend_gpu_layers(hw) if configured_layers < 0 else configured_layers
            return layers, preferred, warnings
        return 0, "cpu", warnings

    # GPU path
    if configured_layers < 0:
        layers = recommend_gpu_layers(hw)
    else:
        layers = configured_layers

    # If recommendation is 0 despite preferred GPU, still try -1 for full offload
    if layers == 0 and preferred in ("cuda", "metal", "rocm"):
        layers = -1

    return layers, preferred, warnings
