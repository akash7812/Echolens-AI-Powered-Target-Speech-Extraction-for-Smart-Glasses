"""
Echolens — Multi-Track Processing Pipelines
===========================================
Pipelines for CHiME-10 Task 3:
  - Track 1: Wearer Own-Speech Enhancement (pipeline.track1)
  - Track 2: Interlocutor Target Speaker Extraction (pipeline.track2)
"""

from .track1 import (
    Track1Pipeline,
    Track1Result,
    Track1PipelineError,
)
from .track2 import (
    Track2Pipeline,
    Track2Result,
    Track2PipelineError,
)

__all__ = [
    "Track1Pipeline",
    "Track1Result",
    "Track1PipelineError",
    "Track2Pipeline",
    "Track2Result",
    "Track2PipelineError",
]
