"""
Echolens — Audio Ingestion and Format Standardization Module
============================================================
Handles audio ingestion from file uploads, raw microphone streams, and in-memory buffers.
Ensures standardized 16,000 Hz mono float32 output required for CHiME-10 Task 3
(both Track 1 Own-Speech Enhancement and Track 2 Target Speaker Extraction / PS4).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import io
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.5
SILENCE_ENERGY_THRESHOLD = 1e-6


class AudioIngestionError(Exception):
    """Raised when audio ingestion fails validation."""
    pass


@dataclass
class IngestionResult:
    """Standardized metadata and waveform returned from audio ingestion."""
    audio: np.ndarray             # 1D float32 mono array at 16000 Hz
    sample_rate: int              # Always 16000
    duration: float               # Duration in seconds
    num_samples: int              # Total sample count
    original_sample_rate: int     # Source sample rate before conversion
    original_channels: int        # Source channel count
    rms_energy: float             # Root mean square energy
    peak_amplitude: float         # Absolute peak amplitude
    saved_path: Optional[str]     # File path if saved to disk
    warnings: list[str]           # List of non-fatal warnings (e.g. clipping, low energy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result metadata to serializable dictionary (omits raw array)."""
        data = asdict(self)
        data.pop("audio", None)
        return data


class AudioIngestion:
    """
    Ingests and standardizes audio from multiple input formats to canonical 16 kHz mono.
    Preserves raw acoustic mixture integrity without applying destructive filters or lossy AGC.
    """

    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    def ingest(
        self,
        source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        input_sample_rate: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> IngestionResult:
        """
        Ingest audio from file path, raw bytes, or in-memory numpy array.

        Args:
            source: File path (str/Path), audio bytes, BytesIO buffer, or numpy array.
            input_sample_rate: Required if source is an in-memory numpy array.
            save_path: Optional path to save the standardized 16 kHz mono WAV.

        Returns:
            IngestionResult containing the standardized audio and metadata.
        """
        warnings: list[str] = []

        # 1. Load raw audio and metadata based on input type
        if isinstance(source, (str, Path)):
            file_path = Path(source)
            if not file_path.is_file():
                raise AudioIngestionError(f"Audio file not found: {file_path}")
            try:
                audio_data, orig_sr = sf.read(file_path, dtype="float32", always_2d=True)
            except Exception as e:
                raise AudioIngestionError(f"Failed to read audio file: {e}") from e

        elif isinstance(source, (bytes, io.BytesIO)):
            buffer = io.BytesIO(source) if isinstance(source, bytes) else source
            try:
                buffer.seek(0)
                audio_data, orig_sr = sf.read(buffer, dtype="float32", always_2d=True)
            except Exception as e:
                # Fallback to torchaudio if libsndfile fails on non-canonical containers
                try:
                    buffer.seek(0)
                    import torchaudio
                    waveform, orig_sr = torchaudio.load(buffer)
                    audio_data = waveform.t().numpy().astype(np.float32)
                except Exception:
                    raise AudioIngestionError(f"Failed to decode audio bytes: {e}") from e

        elif isinstance(source, np.ndarray):
            if input_sample_rate is None or input_sample_rate <= 0:
                raise AudioIngestionError("input_sample_rate must be provided and > 0 for numpy arrays.")
            orig_sr = int(input_sample_rate)
            raw = source.astype(np.float32)
            if raw.ndim == 1:
                audio_data = raw[:, np.newaxis]
            elif raw.ndim == 2:
                # Shape can be (samples, channels) or (channels, samples)
                if raw.shape[0] < raw.shape[1] and raw.shape[0] <= 8:
                    audio_data = raw.T  # Transpose to (samples, channels)
                else:
                    audio_data = raw
            else:
                raise AudioIngestionError(f"Audio array has unsupported shape: {raw.shape}")
        else:
            raise AudioIngestionError(f"Unsupported source type: {type(source)}")

        # 2. Check for empty audio
        if audio_data.size == 0 or audio_data.shape[0] == 0:
            raise AudioIngestionError("Audio is empty (zero samples).")

        # 3. Check for non-finite values (NaN / Inf)
        if not np.all(np.isfinite(audio_data)):
            raise AudioIngestionError("Audio contains invalid non-finite values (NaN or Infinity).")

        samples_count, channels_count = audio_data.shape

        # 4. Multi-channel to mono conversion (preserving relative energy)
        if channels_count > 1:
            mono_audio = np.mean(audio_data, axis=1)
            warnings.append(
                f"Downmixed {channels_count} channels to mono by channel averaging."
            )
        else:
            mono_audio = audio_data[:, 0]

        # 5. Resample to target sample rate (16000 Hz) if needed
        if orig_sr != self.target_sample_rate:
            if orig_sr <= 0:
                raise AudioIngestionError(f"Invalid original sample rate: {orig_sr}")
            gcd_val = math.gcd(orig_sr, self.target_sample_rate)
            up = self.target_sample_rate // gcd_val
            down = orig_sr // gcd_val
            mono_audio = resample_poly(mono_audio, up, down).astype(np.float32)
            warnings.append(
                f"Resampled audio from {orig_sr} Hz to {self.target_sample_rate} Hz."
            )

        duration = float(len(mono_audio) / self.target_sample_rate)

        # 6. Minimum duration validation
        if duration < MIN_DURATION_SECONDS:
            raise AudioIngestionError(
                f"Audio duration ({duration:.2f}s) is shorter than minimum allowed "
                f"({MIN_DURATION_SECONDS}s)."
            )

        # 7. Acoustic sanity metrics
        rms_energy = float(np.sqrt(np.mean(mono_audio.astype(np.float64) ** 2)))
        peak_amp = float(np.max(np.abs(mono_audio)))

        if rms_energy < SILENCE_ENERGY_THRESHOLD:
            warnings.append(
                "Audio appears to be near-silent (RMS energy below threshold)."
            )

        if peak_amp >= 0.999:
            warnings.append(
                "Audio peak reaches digital ceiling (possible clipping/saturation detected)."
            )

        # 8. Save canonical 16 kHz mono WAV file if requested
        saved_file_str = None
        if save_path is not None:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), mono_audio, self.target_sample_rate, subtype="PCM_16")
            saved_file_str = str(out_p.resolve())

        return IngestionResult(
            audio=mono_audio.astype(np.float32),
            sample_rate=self.target_sample_rate,
            duration=duration,
            num_samples=len(mono_audio),
            original_sample_rate=orig_sr,
            original_channels=channels_count,
            rms_energy=rms_energy,
            peak_amplitude=peak_amp,
            saved_path=saved_file_str,
            warnings=warnings,
        )


def ingest_audio(
    source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    input_sample_rate: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> IngestionResult:
    """Convenience helper for AudioIngestion().ingest(...)."""
    return AudioIngestion().ingest(
        source=source,
        input_sample_rate=input_sample_rate,
        save_path=save_path,
    )
