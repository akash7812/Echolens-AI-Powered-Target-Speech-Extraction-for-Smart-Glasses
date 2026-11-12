"""
Echolens — Speaker Diarization and Selection Module
===================================================
Modular speaker diarization package supporting offline spectral AHC diarization,
speaker audio preview generation, and explicit target speaker selection.
"""

from .base import (
    BaseDiarizer,
    SpeakerSegment,
    SpeakerStats,
    DiarizationResult,
)
from .offline_diarizer import SpectralAHCDiarizer
from .preview import (
    SpeakerPreview,
    SpeakerPreviewGenerator,
    SpeakerSelectionManager,
    SpeakerSelectionError,
    MIN_ENROLLMENT_DURATION_WARN,
    MIN_ENROLLMENT_DURATION_ERROR,
    DEFAULT_PREVIEW_DURATION,
)

__all__ = [
    "BaseDiarizer",
    "SpeakerSegment",
    "SpeakerStats",
    "DiarizationResult",
    "SpectralAHCDiarizer",
    "SpeakerPreview",
    "SpeakerPreviewGenerator",
    "SpeakerSelectionManager",
    "SpeakerSelectionError",
    "MIN_ENROLLMENT_DURATION_WARN",
    "MIN_ENROLLMENT_DURATION_ERROR",
    "DEFAULT_PREVIEW_DURATION",
]
