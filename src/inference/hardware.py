"""
Hardware detection for local LLM inference.
Adapted from LocalBot – supports NVIDIA (CUDA), Apple Silicon (Metal),
AMD (ROCm), and CPU fallback.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GPUInfo:
    index: int
    name: str
    vendor: str  # "nvidia" | "apple" | "amd" | "unknown"
    total_vram_mb: int
    free_vram_mb: Optional[int] = None
    backend: str = ""  # "cuda" | "metal" | "rocm" | ""
    shared_memory: bool = False


@dataclass
class HardwareInfo:
    os_name: str
    arch: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    llama_gpu_offload_supported: bool = False
    is_apple_silicon: bool = False

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        order = {"nvidia": 0, "amd": 1, "apple": 2, "unknown": 3}
        return sorted(
            self.gpus, key=lambda g: (order.get(g.vendor, 9), -g.total_vram_mb)
        )[0]

    @property
    def primary_vram_mb(self) -> int:
        g = self.primary_gpu
        return g.total_vram_mb if g else 0

    @property
    def recommended_backend(self) -> str:
        g = self.primary_gpu
        if not g:
            return "cpu"
        return g.backend or "cpu"


def detect_hardware() -> HardwareInfo:
    is_apple = _is_apple_silicon()
    gpus: List[GPUInfo] = []
    gpus.extend(_detect_nvidia_gpus())
    gpus.extend(_detect_amd_gpus())
    if is_apple:
        gpus.extend(_detect_apple_gpus())

    return HardwareInfo(
        os_name=platform.system(),
        arch=platform.machine(),
        cpu_name=_cpu_name(),
        physical_cores=_physical_cores(),
        logical_cores=_logical_cores(),
        total_ram_gb=_total_ram_gb(),
        gpus=gpus,
        llama_gpu_offload_supported=_llama_supports_gpu_offload(),
        is_apple_silicon=is_apple,
    )


def recommend_gpu_layers(
    hardware: HardwareInfo,
    model_size_mb: Optional[float] = None,
    prefer_full_offload: bool = True,
) -> int:
    """0 = CPU only, -1 = offload as many layers as possible."""
    if not hardware.has_gpu:
        return 0
    if not hardware.llama_gpu_offload_supported and hardware.recommended_backend != "metal":
        return 0

    g = hardware.primary_gpu
    if g is None:
        return 0

    if g.vendor == "apple":
        return -1 if prefer_full_offload else 0

    vram = g.total_vram_mb
    if vram >= 2000:
        return -1
    return 0


def recommend_context_size(hardware: HardwareInfo, default: int = 8192) -> int:
    ram = hardware.total_ram_gb
    vram = hardware.primary_vram_mb
    g = hardware.primary_gpu

    if g and g.shared_memory:
        if ram < 8:
            return min(default, 2048)
        if ram < 16:
            return min(default, 4096)
        if ram < 24:
            return min(default, 8192)
        return default

    if ram < 8 and vram < 4000:
        return min(default, 2048)
    if ram < 12 or vram < 6000:
        return min(default, 4096)
    return default


def format_hardware_summary(h: HardwareInfo) -> str:
    lines = [
        f"OS: {h.os_name} ({h.arch})",
        f"CPU: {h.cpu_name}",
        f"Cores: {h.physical_cores} physical / {h.logical_cores} logical",
        f"RAM: {h.total_ram_gb:.1f} GB",
        f"Apple Silicon: {h.is_apple_silicon}",
        f"llama.cpp GPU offload supported: {h.llama_gpu_offload_supported}",
        f"Recommended backend: {h.recommended_backend}",
    ]
    if h.gpus:
        for g in h.gpus:
            free = f", free {g.free_vram_mb} MB" if g.free_vram_mb is not None else ""
            shared = " [unified memory]" if g.shared_memory else ""
            lines.append(
                f"GPU {g.index}: {g.name} ({g.vendor}/{g.backend}) — "
                f"{g.total_vram_mb} MB{free}{shared}"
            )
    else:
        lines.append("GPU: none detected")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cpu_name() -> str:
    try:
        import cpuinfo
        brand = cpuinfo.get_cpu_info().get("brand_raw")
        if brand:
            return brand
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3
            ).strip()
            if out:
                return out
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown"


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return _logical_cores()


def _logical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=True)
        if n:
            return n
    except Exception:
        pass
    return os.cpu_count() or 1


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ).strip()
            return int(out) / (1024 ** 3)
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _is_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return True
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm64"], text=True, timeout=2
        ).strip()
        return out == "1"
    except Exception:
        return False


def _detect_apple_gpus() -> List[GPUInfo]:
    if not _is_apple_silicon():
        return []
    chip = "Apple Silicon"
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3
        ).strip()
        if out:
            chip = out
    except Exception:
        pass
    ram_gb = _total_ram_gb()
    usable_mb = int(max(0, (ram_gb - 2.0) * 1024))
    return [
        GPUInfo(
            index=0,
            name=chip,
            vendor="apple",
            total_vram_mb=usable_mb,
            free_vram_mb=None,
            backend="metal",
            shared_memory=True,
        )
    ]


def _detect_nvidia_gpus() -> List[GPUInfo]:
    gpus = _gpus_via_nvml()
    if gpus:
        return gpus
    return _gpus_via_nvidia_smi()


def _gpus_via_nvml() -> List[GPUInfo]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result.append(
                GPUInfo(
                    index=i,
                    name=name,
                    vendor="nvidia",
                    total_vram_mb=int(mem.total / (1024 * 1024)),
                    free_vram_mb=int(mem.free / (1024 * 1024)),
                    backend="cuda",
                    shared_memory=False,
                )
            )
        pynvml.nvmlShutdown()
        return result
    except Exception:
        return []


def _gpus_via_nvidia_smi() -> List[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        result = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            result.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor="nvidia",
                    total_vram_mb=int(float(parts[2])),
                    free_vram_mb=int(float(parts[3])),
                    backend="cuda",
                    shared_memory=False,
                )
            )
        return result
    except Exception:
        return []


def _detect_amd_gpus() -> List[GPUInfo]:
    # Simplified – full ROCm detection can be expanded later
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--csv"],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
        result = []
        for i, line in enumerate(out.strip().splitlines()[1:]):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result.append(
                    GPUInfo(
                        index=i,
                        name=parts[-1],
                        vendor="amd",
                        total_vram_mb=0,  # would need extra call
                        backend="rocm",
                        shared_memory=False,
                    )
                )
        return result
    except Exception:
        return []


def _llama_supports_gpu_offload() -> bool:
    """Best-effort check whether the installed llama-cpp-python has GPU support."""
    try:
        from llama_cpp import llama_supports_gpu_offload
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False
