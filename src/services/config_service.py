"""Persist UI settings to .env so they survive restarts."""
from __future__ import annotations

from src.config import ENV_FILE, settings


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def write_env_value(key: str, value: str) -> None:
    env = read_env()
    if value == "" and key in env:
        del env[key]
    else:
        env[key] = value
    lines = [f"{k}={v}" for k, v in env.items()]
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _set_setting(name: str, value) -> None:
    try:
        object.__setattr__(settings, name, value)
    except Exception:
        pass


def set_model_path(path: str) -> str:
    path = path.strip().replace("\\", "/")
    write_env_value("LLM_MODEL_PATH", path)
    _set_setting("llm_model_path", path)
    return path


def get_model_path() -> str:
    env = read_env()
    return env.get("LLM_MODEL_PATH") or getattr(settings, "llm_model_path", "") or ""


def set_gpu_layers(layers: int) -> int:
    write_env_value("LLM_GPU_LAYERS", str(int(layers)))
    _set_setting("llm_gpu_layers", int(layers))
    return int(layers)


def get_gpu_layers() -> int:
    env = read_env()
    raw = env.get("LLM_GPU_LAYERS")
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            pass
    return int(getattr(settings, "llm_gpu_layers", -1))


def set_preferred_backend(backend: str) -> str:
    backend = (backend or "auto").lower().strip()
    write_env_value("LLM_PREFERRED_BACKEND", backend)
    _set_setting("llm_preferred_backend", backend)
    return backend


def get_preferred_backend() -> str:
    env = read_env()
    return (
        env.get("LLM_PREFERRED_BACKEND")
        or getattr(settings, "llm_preferred_backend", "auto")
        or "auto"
    ).lower()
