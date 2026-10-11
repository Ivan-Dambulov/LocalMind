"""
Detect GPU backend and install a matching llama-cpp-python build.

Backends:
  - cuda  : NVIDIA (prebuilt wheels when possible)
  - metal : Apple Silicon (build with GGML_METAL)
  - rocm  : AMD (build with GGML_HIPBLAS / ROCm)
  - vulkan: cross-vendor fallback (build with GGML_VULKAN) — useful on some Intel/AMD
  - cpu   : CPU-only wheel or plain install
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from src.inference.hardware import detect_hardware, HardwareInfo


WHEEL_BASE = "https://abetlen.github.io/llama-cpp-python/whl"


@dataclass
class BackendPlan:
    backend: str
    label: str
    available: bool
    reason: str
    install_mode: str  # "wheel" | "source" | "cpu"
    pip_args: list[str]
    env: dict[str, str]
    notes: str


def detect_cuda_version() -> Optional[str]:
    """Return major.minor from nvidia-smi if possible (driver-reported CUDA)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=8, stderr=subprocess.DEVNULL
        )
        # e.g. "CUDA Version: 12.4"
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return None


def cuda_wheel_tag(cuda_version: Optional[str]) -> str:
    """
    Map driver CUDA version to a published wheel tag.
    Prefer safer common tags; fall back to cu121.
    """
    if not cuda_version:
        return "cu121"
    try:
        major, minor = cuda_version.split(".")
        major_i, minor_i = int(major), int(minor)
    except Exception:
        return "cu121"

    # Prefer wheels that are widely published
    if major_i > 12 or (major_i == 12 and minor_i >= 4):
        return "cu124"
    if major_i == 12 and minor_i >= 2:
        return "cu122"
    if major_i == 12:
        return "cu121"
    if major_i == 11:
        return "cu118"
    return "cu121"


def current_llama_info() -> dict:
    info = {
        "installed": False,
        "version": None,
        "gpu_offload_supported": False,
        "error": None,
    }
    try:
        import llama_cpp
        info["installed"] = True
        info["version"] = getattr(llama_cpp, "__version__", "unknown")
        try:
            from llama_cpp import llama_supports_gpu_offload
            info["gpu_offload_supported"] = bool(llama_supports_gpu_offload())
        except Exception:
            # Older builds may not expose this
            info["gpu_offload_supported"] = False
    except Exception as e:
        info["error"] = str(e)
    return info


def build_plans(hw: HardwareInfo | None = None) -> list[BackendPlan]:
    hw = hw or detect_hardware()
    system = platform.system().lower()
    plans: list[BackendPlan] = []

    has_cuda = any(g.backend == "cuda" for g in hw.gpus)
    has_metal = any(g.backend == "metal" for g in hw.gpus) or hw.is_apple_silicon
    has_rocm = any(g.backend == "rocm" for g in hw.gpus)

    # --- CUDA ---
    cuda_ver = detect_cuda_version() if has_cuda else None
    tag = cuda_wheel_tag(cuda_ver)
    cuda_env = {
        "CMAKE_ARGS": "-DGGML_CUDA=on",
        "FORCE_CMAKE": "1",
    }
    if has_cuda:
        # Prefer prebuilt wheel
        plans.append(
            BackendPlan(
                backend="cuda",
                label=f"NVIDIA CUDA (wheel {tag})",
                available=True,
                reason=f"NVIDIA GPU detected" + (f", driver CUDA {cuda_ver}" if cuda_ver else ""),
                install_mode="wheel",
                pip_args=[
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "llama-cpp-python",
                    "--extra-index-url",
                    f"{WHEEL_BASE}/{tag}",
                ],
                env={},
                notes=(
                    f"Installs prebuilt llama-cpp-python with {tag}. "
                    "Requires a compatible NVIDIA driver. "
                    "If the wheel fails, use 'CUDA from source' below."
                ),
            )
        )
        plans.append(
            BackendPlan(
                backend="cuda_source",
                label="NVIDIA CUDA (build from source)",
                available=True,
                reason="Needs CUDA Toolkit + compiler (Visual Studio on Windows, gcc on Linux)",
                install_mode="source",
                pip_args=[
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "llama-cpp-python",
                ],
                env=cuda_env,
                notes="Slower install. Use when wheels don't match your Python/OS/CUDA.",
            )
        )
    else:
        plans.append(
            BackendPlan(
                backend="cuda",
                label="NVIDIA CUDA",
                available=False,
                reason="No NVIDIA GPU detected",
                install_mode="wheel",
                pip_args=[],
                env={},
                notes="",
            )
        )

    # --- Metal ---
    if has_metal and system == "darwin":
        plans.append(
            BackendPlan(
                backend="metal",
                label="Apple Metal",
                available=True,
                reason="Apple Silicon detected",
                install_mode="source",
                pip_args=[
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "llama-cpp-python",
                ],
                env={
                    "CMAKE_ARGS": "-DGGML_METAL=on",
                    "FORCE_CMAKE": "1",
                },
                notes="Builds llama-cpp-python with Metal. Xcode command-line tools required.",
            )
        )
    else:
        plans.append(
            BackendPlan(
                backend="metal",
                label="Apple Metal",
                available=system == "darwin" and has_metal,
                reason="Only on Apple Silicon macOS" if system != "darwin" else "Not detected",
                install_mode="source",
                pip_args=[],
                env={},
                notes="",
            )
        )

    # --- ROCm ---
    if has_rocm:
        plans.append(
            BackendPlan(
                backend="rocm",
                label="AMD ROCm",
                available=True,
                reason="AMD GPU / ROCm tools detected",
                install_mode="source",
                pip_args=[
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "llama-cpp-python",
                ],
                env={
                    "CMAKE_ARGS": "-DGGML_HIPBLAS=on",
                    "FORCE_CMAKE": "1",
                },
                notes="Requires ROCm toolkit. Build times can be long.",
            )
        )
    else:
        plans.append(
            BackendPlan(
                backend="rocm",
                label="AMD ROCm",
                available=False,
                reason="No ROCm/AMD GPU detected",
                install_mode="source",
                pip_args=[],
                env={},
                notes="",
            )
        )

    # --- Vulkan (optional Intel/AMD/others) ---
    plans.append(
        BackendPlan(
            backend="vulkan",
            label="Vulkan (experimental)",
            available=True,
            reason="Optional cross-vendor GPU path (Intel / AMD / NVIDIA)",
            install_mode="source",
            pip_args=[
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                "llama-cpp-python",
            ],
            env={
                "CMAKE_ARGS": "-DGGML_VULKAN=on",
                "FORCE_CMAKE": "1",
            },
            notes="Needs Vulkan SDK. Use when CUDA/ROCm/Metal are not an option.",
        )
    )

    # --- CPU ---
    plans.append(
        BackendPlan(
            backend="cpu",
            label="CPU only",
            available=True,
            reason="Works on any machine",
            install_mode="wheel",
            pip_args=[
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                "llama-cpp-python",
                "--extra-index-url",
                f"{WHEEL_BASE}/cpu",
            ],
            env={},
            notes="Fastest install, slowest inference. Good fallback.",
        )
    )

    return plans


def recommended_plan(plans: list[BackendPlan] | None = None) -> BackendPlan:
    plans = plans or build_plans()
    for key in ("cuda", "metal", "rocm", "cpu"):
        for p in plans:
            if p.backend == key and p.available and p.pip_args:
                return p
    return plans[-1]


def install_backend(plan: BackendPlan, timeout: int = 1800) -> tuple[bool, str]:
    """
    Run pip install for the given plan in the current interpreter.
    Returns (success, log_text).
    """
    if not plan.pip_args:
        return False, f"Backend {plan.backend} is not available on this machine."

    cmd = [sys.executable, "-m", "pip", *plan.pip_args]
    env = os.environ.copy()
    env.update(plan.env)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return False, log[-8000:]
        return True, log[-8000:]
    except subprocess.TimeoutExpired:
        return False, "Install timed out. Try running the pip command manually in a terminal."
    except Exception as e:
        return False, str(e)


def format_pip_command(plan: BackendPlan) -> str:
    """Human-readable command for copy-paste."""
    if not plan.pip_args:
        return "# Not available"
    parts = []
    if plan.env.get("CMAKE_ARGS"):
        if platform.system() == "Windows":
            parts.append(f'$env:CMAKE_ARGS="{plan.env["CMAKE_ARGS"]}"')
            if plan.env.get("FORCE_CMAKE"):
                parts.append(f'$env:FORCE_CMAKE="{plan.env["FORCE_CMAKE"]}"')
            parts.append(f"python -m pip {' '.join(plan.pip_args)}")
            return "\n".join(parts)
        export = " ".join(f'{k}="{v}"' for k, v in plan.env.items())
        return f"{export} python -m pip {' '.join(plan.pip_args)}"
    return f"python -m pip {' '.join(plan.pip_args)}"
