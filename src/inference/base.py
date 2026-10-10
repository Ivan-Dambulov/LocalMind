"""Abstract base class for LLM runtimes (copied & adapted from LocalBot)."""
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        raise NotImplementedError
