"""
Local GGUF runtime using llama-cpp-python.
Adapted from LocalBot.
"""
from __future__ import annotations

from typing import Generator, List, Dict

from llama_cpp import Llama

from src.inference.base import BaseLLM


class LlamaRuntime(BaseLLM):
    def __init__(
        self,
        model_path: str,
        context_size: int = 8192,
        gpu_layers: int = 0,
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.temperature = temperature

        print(f"[LlamaRuntime] Loading model: {model_path}")
        print(f"[LlamaRuntime] Context size : {context_size}")
        print(f"[LlamaRuntime] GPU layers   : {gpu_layers}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        print("[LlamaRuntime] Model loaded.")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        result = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            stream=False,
        )
        return result["choices"][0]["message"]["content"]

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
