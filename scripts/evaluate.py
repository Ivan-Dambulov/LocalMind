#!/usr/bin/env python3
"""Placeholder evaluation script – expand with golden set + RAGAS."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("Evaluation harness – coming in Phase 2")
print("Plan:")
print("  1. Load data/golden_set.json")
print("  2. Run each question through the agent")
print("  3. Score with RAGAS (faithfulness, answer_relevancy, context_precision)")
print("  4. Print summary table + save results")
print("\nCreate data/golden_set.json with entries like:")
print('''
[
  {
    "id": "q1",
    "question": "What is the main purpose of LocalMind?",
    "ground_truth": "A privacy-first Agentic RAG knowledge assistant...",
    "relevant_sources": ["readme.md"]
  }
]
''')
