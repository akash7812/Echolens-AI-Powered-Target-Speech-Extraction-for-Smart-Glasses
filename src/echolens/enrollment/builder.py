"""
Echolens — Target Speaker Enrollment Audio Builder
==================================================
Assembles high-quality 16 kHz mono enrollment audio from speech segments belonging
exclusively to the user-confirmed target speaker.

Strict CHiME-10 Track 2 constraints:
  - Collects only speech segments belonging to the confirmed speaker.
  - Leaves the original mixture untouched.
  - Maintains strict 16,000 Hz mono compatibility for PS4.
  - Enforces minimum duration checks and issues warnings for sparse speech turns.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import soundfile as sf

from echolens.diarization.base import DiarizationResult, SpeakerSegment
from echolens.diarization.preview import (
    MIN_ENROLLMENT_DURATION_WARN,
    MIN_ENROLLMENT_DURATION_ERROR,
)


TARGET_SAMPLE_RATE = 16000
DEFAULT_MAX_ENROLLMENT_DURATION = 5.0  # Max duration in seconds to optimize PS4 memory
MIN_SEGMENT_DURATION = 0.2             # Ignore segments shorter than 200ms


class EnrollmentError(Exception):
    """Raised when enrollment generation fails validation."""
    pass


@dataclass
class EnrollmentResult:
    """Standardized metadata and waveform for the generated target enrollment."""
    audio: np.ndarray             # 1D float32 mono array at 16000 Hz
    duration: float               # Duration in seconds
    speaker: str                  # Confirmed speaker label (e.g. "Speaker B")
    sample_rate: int              # Always 16000 Hz
    num_segments_used: int        # Number of dialogue segments concatenated
    rms_energy: float             # Root mean square energy
    peak_amplitude: float         # Peak absolute amplitude
    saved_path: Optional[str]     # File path if written to disk
    warnings: List[str]           # Quality warnings (e.g. short duration, low energy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary (excluding raw audio array)."""
        data = asdict(self)
        data.pop("audio", None)
        return data


class EnrollmentBuilder:
    """
    Constructs and validates a 16 kHz mono enrollment WAV file from diarized speech turns.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_enrollment_duration: float = DEFAULT_MAX_ENROLLMENT_DURATION,
        min_segment_duration: float = MIN_SEGMENT_DURATION,
    ):
        self.sample_rate = sample_rate
        self.max_enrollment_duration = max_enrollment_duration
        self.min_segment_duration = min_segment_duration

    def _crossfade_concat(
        self,
        chunks: List[np.ndarray],
        fade_samples: int = 80,  # 5ms crossfade at 16 kHz
    ) -> np.ndarray:
        """Concatenate audio chunks with short cosine cross-fades to prevent phase clicks."""
        if not chunks:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0].copy()

        fade_in = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        fade_out = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2

        out_chunks = []
        for i, chunk in enumerate(chunks):
            c = chunk.copy()
            if len(c) > 2 * fade_samples:
                c[:fade_samples] *= fade_in
                c[-fade_samples:] *= fade_out
            out_chunks.append(c)

        return np.concatenate(out_chunks).astype(np.float32)

    def build_enrollment(
        self,
        audio: np.ndarray,
        diarization_result: DiarizationResult,
        selected_speaker: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> EnrollmentResult:
        """
        Build a clean enrollment audio signal for the selected speaker.

        Args:
            audio: 1D float32 mono mixture audio array.
            diarization_result: Diarization output containing speaker segments.
            selected_speaker: Confirmed speaker label (e.g. "Speaker B").
            save_path: Optional path to save the enrollment WAV file.

        Returns:
            EnrollmentResult containing standardized 16 kHz mono enrollment audio and metadata.
        """
        warnings: List[str] = []

        if selected_speaker not in diarization_result.speakers:
            raise EnrollmentError(
                f"Selected speaker '{selected_speaker}' not found in detected speakers: "
                f"{diarization_result.speakers}"
            )

        # 1. Gather all segments for this speaker
        speaker_segments = [
            s for s in diarization_result.segments
            if s.speaker == selected_speaker and s.duration >= self.min_segment_duration
        ]

        if not speaker_segments:
            raise EnrollmentError(
                f"No usable speech segments found for {selected_speaker} "
                f"(segments must be >= {self.min_segment_duration}s)."
            )

        # 2. Prioritize longer continuous turns first for best acoustic quality
        sorted_segments = sorted(speaker_segments, key=lambda s: s.duration, reverse=True)

        max_samples = int(self.max_enrollment_duration * self.sample_rate)
        collected_chunks: List[np.ndarray] = []
        collected_samples = 0
        used_segment_count = 0

        for seg in sorted_segments:
            start_idx = int(seg.start_time * self.sample_rate)
            end_idx = min(len(audio), int(seg.end_time * self.sample_rate))
            if end_idx <= start_idx:
                continue

            chunk = audio[start_idx:end_idx].copy()
            needed = max_samples - collected_samples

            if len(chunk) > needed:
                chunk = chunk[:needed]

            collected_chunks.append(chunk)
            collected_samples += len(chunk)
            used_segment_count += 1

            if collected_samples >= max_samples:
                break

        # 3. Concatenate chunks with smoothing
        enrollment_audio = self._crossfade_concat(collected_chunks)
        duration = round(float(len(enrollment_audio) / self.sample_rate), 3)

        # 4. Critical & Warning Duration Validations
        if duration < MIN_ENROLLMENT_DURATION_ERROR:
            raise EnrollmentError(
                f"Insufficient speech for enrollment: {selected_speaker} has only {duration:.2f}s "
                f"of speech. PS4 model requires at least {MIN_ENROLLMENT_DURATION_ERROR}s."
            )

        if duration < MIN_ENROLLMENT_DURATION_WARN:
            warnings.append(
                f"Enrollment audio is relatively short ({duration:.2f}s). "
                f"PS4 extraction performs best with at least {MIN_ENROLLMENT_DURATION_WARN}s."
            )

        # 5. Sanity checks (NaN, Inf, Silence)
        if not np.all(np.isfinite(enrollment_audio)):
            raise EnrollmentError("Enrollment audio contains invalid NaN or Infinite values.")

        rms = float(np.sqrt(np.mean(enrollment_audio.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(enrollment_audio)))

        if rms < 1e-5:
            raise EnrollmentError(
                "Enrollment audio is near-silent (RMS energy < 1e-5). Cannot enroll target speaker."
            )

        # 6. Save to disk if requested
        saved_file = None
        if save_path is not None:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), enrollment_audio, self.sample_rate, subtype="PCM_16")
            saved_file = str(out_p.resolve())

        return EnrollmentResult(
            audio=enrollment_audio,
            duration=duration,
            speaker=selected_speaker,
            sample_rate=self.sample_rate,
            num_segments_used=used_segment_count,
            rms_energy=rms,
            peak_amplitude=peak,
            saved_path=saved_file,
            warnings=warnings,
        )


def build_enrollment(
    audio: np.ndarray,
    diarization_result: DiarizationResult,
    selected_speaker: str,
    save_path: Optional[Union[str, Path]] = None,
) -> EnrollmentResult:
    """Convenience helper for EnrollmentBuilder().build_enrollment(...)."""
    return EnrollmentBuilder().build_enrollment(
        audio=audio,
        diarization_result=diarization_result,
        selected_speaker=selected_speaker,
        save_path=save_path,
    )
