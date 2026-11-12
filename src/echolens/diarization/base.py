"""
Echolens — Speaker Diarization Base Definitions
================================================
Defines segment data structures and abstract base class for diarizers.
Supports modular backends (default: SpectralAHCDiarizer; optional: PyannoteDiarizer).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SpeakerSegment:
    """Represents a continuous segment of speech spoken by a single speaker."""
    speaker: str          # e.g. "Speaker A", "Speaker B"
    start_time: float     # Start timestamp in seconds
    end_time: float       # End timestamp in seconds
    duration: float       # Duration in seconds
    confidence: float     # Estimated confidence score (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerStats:
    """Aggregated statistics for a detected speaker across the recording."""
    speaker: str
    total_duration: float
    num_segments: float
    percentage: float
    avg_segment_duration: float


@dataclass
class DiarizationResult:
    """Complete result returned from speaker diarization."""
    segments: List[SpeakerSegment]
    speakers: List[str]
    speaker_stats: Dict[str, SpeakerStats]
    total_audio_duration: float
    total_speech_duration: float
    sample_rate: int
    num_speakers: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
            "total_audio_duration": self.total_audio_duration,
            "total_speech_duration": self.total_speech_duration,
            "sample_rate": self.sample_rate,
            "segments": [s.to_dict() for s in self.segments],
            "speaker_stats": {
                k: asdict(v) for k, v in self.speaker_stats.items()
            },
        }


class BaseDiarizer(ABC):
    """Abstract interface for speaker diarization backends."""

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Diarize 16 kHz mono audio and return timestamped speaker segments.

        Args:
            audio: 1D float32 mono audio array.
            sample_rate: Audio sample rate (must be 16000).
            num_speakers: Optional expected number of speakers.

        Returns:
            DiarizationResult containing detected speakers and speech segments.
        """
        pass
