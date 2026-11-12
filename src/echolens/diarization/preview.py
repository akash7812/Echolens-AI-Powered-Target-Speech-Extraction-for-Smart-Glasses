"""
Echolens — Speaker Preview and Manual Selection Module
======================================================
Handles generation of auditory preview snippets for detected speakers and manages
explicit target selection by the user.

Strict CHiME-10 Track 2 constraints:
  - Never automatically assumes Speaker A is the target.
  - Keeps diarization and target selection strictly separate.
  - Enforces explicit user confirmation before enrollment.
  - Analyzes speech duration and alerts user if speech is insufficient for enrollment.
  - Allows previewing, switching, and re-confirming selections.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import soundfile as sf

from .base import DiarizationResult, SpeakerSegment


MIN_ENROLLMENT_DURATION_WARN = 1.5   # Warning threshold in seconds
MIN_ENROLLMENT_DURATION_ERROR = 0.5  # Critical rejection threshold in seconds
DEFAULT_PREVIEW_DURATION = 3.0      # Target preview clip duration in seconds


class SpeakerSelectionError(Exception):
    """Raised when speaker selection or confirmation fails validation."""
    pass


@dataclass
class SpeakerPreview:
    """Auditory preview snippet and metadata for a detected speaker."""
    speaker: str                          # e.g. "Speaker A", "Speaker B"
    preview_audio: np.ndarray             # 1D float32 audio snippet
    duration: float                       # Duration of preview clip in seconds
    sample_rate: int                      # Sample rate (16000 Hz)
    total_detected_duration: float        # Total detected dialogue in the mixture
    segment_count: int                    # Number of detected turns
    is_sufficient_for_enrollment: bool    # True if >= 1.5s
    quality_warning: Optional[str]        # Non-fatal warning if duration is low
    saved_path: Optional[str]             # Path to WAV preview file on disk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excluding raw audio array)."""
        data = asdict(self)
        data.pop("preview_audio", None)
        return data


class SpeakerPreviewGenerator:
    """
    Extracts clean, non-overlapping preview clips for each detected speaker
    and validates speech duration for downstream PS4 enrollment.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_preview_duration: float = DEFAULT_PREVIEW_DURATION,
    ):
        self.sample_rate = sample_rate
        self.target_preview_duration = target_preview_duration

    def _apply_fade(self, audio: np.ndarray, fade_samples: int = 160) -> np.ndarray:
        """Apply 10ms cosine fade-in and fade-out to prevent playback clicks."""
        if len(audio) <= 2 * fade_samples:
            return audio.copy()
        out = audio.copy()
        fade_in = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        fade_out = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        out[:fade_samples] *= fade_in
        out[-fade_samples:] *= fade_out
        return out

    def generate_previews(
        self,
        audio: np.ndarray,
        diarization_result: DiarizationResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, SpeakerPreview]:
        """
        Generate audio preview clips for all detected speakers.

        Args:
            audio: 1D float32 mono mixture audio array.
            diarization_result: Output from Diarization engine.
            output_dir: Optional directory to save preview WAV files.

        Returns:
            Dictionary mapping speaker name ("Speaker A", "Speaker B", ...) to SpeakerPreview.
        """
        previews: Dict[str, SpeakerPreview] = {}

        if diarization_result.num_speakers == 0:
            return previews

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
        else:
            out_p = None

        target_preview_samples = int(self.target_preview_duration * self.sample_rate)

        for speaker in diarization_result.speakers:
            # Collect all segments belonging to this speaker
            speaker_segs = [
                s for s in diarization_result.segments if s.speaker == speaker
            ]
            total_duration = sum(s.duration for s in speaker_segs)

            # Sort segments by duration descending to pick the cleanest, longest turns first
            sorted_segs = sorted(speaker_segs, key=lambda s: s.duration, reverse=True)

            preview_chunks: List[np.ndarray] = []
            collected_samples = 0

            for seg in sorted_segs:
                start_samp = int(seg.start_time * self.sample_rate)
                end_samp = min(len(audio), int(seg.end_time * self.sample_rate))
                if end_samp <= start_samp:
                    continue

                chunk = audio[start_samp:end_samp]
                needed = target_preview_samples - collected_samples
                if len(chunk) > needed and collected_samples > 0:
                    chunk = chunk[:needed]

                preview_chunks.append(chunk)
                collected_samples += len(chunk)
                if collected_samples >= target_preview_samples:
                    break

            if preview_chunks:
                preview_audio = np.concatenate(preview_chunks).astype(np.float32)
            else:
                preview_audio = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)

            # Apply fade to preview
            preview_audio = self._apply_fade(preview_audio)
            actual_duration = round(float(len(preview_audio) / self.sample_rate), 3)

            # Check sufficiency for downstream PS4 enrollment
            if total_duration < MIN_ENROLLMENT_DURATION_ERROR:
                is_sufficient = False
                quality_warning = (
                    f"Critical: {speaker} only has {total_duration:.2f}s of detected speech. "
                    f"Minimum required is {MIN_ENROLLMENT_DURATION_ERROR}s. Cannot be used for enrollment."
                )
            elif total_duration < MIN_ENROLLMENT_DURATION_WARN:
                is_sufficient = False
                quality_warning = (
                    f"Warning: {speaker} has {total_duration:.2f}s of detected speech. "
                    f"PS4 extraction performs best with at least {MIN_ENROLLMENT_DURATION_WARN}s of clean enrollment."
                )
            else:
                is_sufficient = True
                quality_warning = None

            saved_file = None
            if out_p is not None:
                slug = speaker.lower().replace(" ", "_")
                dest = out_p / f"preview_{slug}.wav"
                sf.write(str(dest), preview_audio, self.sample_rate, subtype="PCM_16")
                saved_file = str(dest.resolve())

            previews[speaker] = SpeakerPreview(
                speaker=speaker,
                preview_audio=preview_audio,
                duration=actual_duration,
                sample_rate=self.sample_rate,
                total_detected_duration=round(total_duration, 3),
                segment_count=len(speaker_segs),
                is_sufficient_for_enrollment=is_sufficient,
                quality_warning=quality_warning,
                saved_path=saved_file,
            )

        return previews


class SpeakerSelectionManager:
    """
    Manages the user's manual target speaker selection workflow.
    Guarantees that no default target is assumed, requiring explicit user confirmation.
    """

    def __init__(self, available_speakers: Optional[List[str]] = None):
        self.available_speakers: List[str] = available_speakers or []
        self._selected_speaker: Optional[str] = None
        self._confirmed: bool = False

    def update_available_speakers(self, speakers: List[str]) -> None:
        """Update available speakers from a new diarization pass and reset selection."""
        self.available_speakers = list(speakers)
        self._selected_speaker = None
        self._confirmed = False

    def select(self, speaker: str) -> None:
        """
        Choose a target speaker from the available detected speakers.
        Does not confirm automatically — user must explicitly call confirm().
        """
        if not self.available_speakers:
            raise SpeakerSelectionError("No speakers have been detected yet.")
        if speaker not in self.available_speakers:
            raise SpeakerSelectionError(
                f"Selected speaker '{speaker}' is not in detected list: {self.available_speakers}"
            )
        self._selected_speaker = speaker
        self._confirmed = False  # Selection changed, requires re-confirmation

    def confirm(self) -> str:
        """
        Explicitly confirm the selected speaker as the intended target.
        Returns the confirmed speaker name.
        """
        if self._selected_speaker is None:
            raise SpeakerSelectionError(
                "Cannot confirm target speaker: No speaker has been selected. "
                "The user must explicitly preview and select a speaker."
            )
        self._confirmed = True
        return self._selected_speaker

    def reset(self) -> None:
        """Reset the selection state to allow the user to choose another speaker."""
        self._selected_speaker = None
        self._confirmed = False

    @property
    def selected_speaker(self) -> Optional[str]:
        return self._selected_speaker

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize selection state for UI binding."""
        return {
            "available_speakers": self.available_speakers,
            "selected_speaker": self._selected_speaker,
            "is_confirmed": self._confirmed,
        }
