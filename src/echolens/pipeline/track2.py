"""
Echolens — Track 2: Interlocutor Target Speaker Extraction Pipeline
===================================================================
Coordinates the complete Track 2 workflow:
  1. Validates and ingests 16 kHz mono conversation mixture.
  2. Builds enrollment audio for the user-selected speaker.
  3. Executes offline inference via the existing PS4 model (checkpoint_epoch037.pt).
  4. Saves extracted audio and calculates SI-SDR metrics (only if reference target exists).

Preserves the existing Member 2 PS4 model and evaluation integration untouched.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import time

import numpy as np
import soundfile as sf

from echolens.diarization.base import DiarizationResult
from echolens.enrollment.builder import EnrollmentBuilder, EnrollmentResult
from echolens.models.ps4_extractor import PS4Extractor
from echolens.models.evaluation import evaluate_sample


import os

def _resolve_default_checkpoint() -> str:
    candidates = [
        os.environ.get("ECHOLENS_CHECKPOINT_PATH"),
        r"C:\Echolens_Models\checkpoint_finetuned.pt",
        r"C:\Echolens_Models\checkpoint_epoch037.pt",
        "models/checkpoint_finetuned.pt",
        "models/checkpoint_epoch037.pt",
        "/app/models/checkpoint_finetuned.pt",
        "/app/models/checkpoint_epoch037.pt",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return str(Path("models/checkpoint_finetuned.pt"))


DEFAULT_CHECKPOINT_PATH = _resolve_default_checkpoint()


class Track2PipelineError(Exception):
    """Raised when Track 2 pipeline execution encounters a critical error."""
    pass


@dataclass
class Track2Result:
    """Complete output produced by Track 2 Target Speaker Extraction."""
    extracted_audio: np.ndarray             # 1D float32 extracted target speech at 16000 Hz
    mixture_audio: np.ndarray               # 1D float32 original untouched mixture at 16000 Hz
    enrollment_audio: np.ndarray            # 1D float32 enrollment audio at 16000 Hz
    selected_speaker: str                   # Label of the confirmed interlocutor
    sample_rate: int                        # Always 16000 Hz
    duration: float                         # Audio duration in seconds
    output_path: Optional[str]              # Saved path to extracted WAV file
    enrollment_path: Optional[str]          # Saved path to enrollment WAV file
    metrics: Optional[Dict[str, float]]     # SI-SDR metrics (None if no clean reference provided)
    warnings: List[str]                     # Pipeline warnings
    execution_time: float                   # Processing runtime in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary (omits raw arrays)."""
        data = asdict(self)
        data.pop("extracted_audio", None)
        data.pop("mixture_audio", None)
        data.pop("enrollment_audio", None)
        return data


class Track2Pipeline:
    """
    End-to-end pipeline for CHiME-10 Track 2: Specified Interlocutor Extraction.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.extractor: Optional[PS4Extractor] = None
        self.enrollment_builder = EnrollmentBuilder(sample_rate=16000)

    def _ensure_model_loaded(self) -> PS4Extractor:
        """Lazy load the PS4 extractor on first inference request."""
        if self.extractor is None:
            if not self.checkpoint_path.is_file():
                raise Track2PipelineError(
                    f"PS4 checkpoint file not found at: {self.checkpoint_path}"
                )
            self.extractor = PS4Extractor(
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
            )
        return self.extractor

    def extract(
        self,
        mixture: np.ndarray,
        diarization_result: Optional[DiarizationResult] = None,
        selected_speaker: Optional[str] = None,
        enrollment_audio: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
        clean_target: Optional[np.ndarray] = None,
    ) -> Track2Result:
        """
        Extract the target speaker from a conversation mixture.

        Args:
            mixture: 1D float32 mono mixture array at 16000 Hz.
            diarization_result: Output from Diarization engine (if extracting via speaker selection).
            selected_speaker: Confirmed speaker label (required if using diarization_result).
            enrollment_audio: Direct enrollment array (alternative to diarization_result).
            output_path: Optional destination to save the extracted audio WAV.
            clean_target: Optional ground-truth clean audio for SI-SDR evaluation.

        Returns:
            Track2Result containing extracted speech, audio metadata, and metrics.
        """
        start_time = time.time()
        warnings: List[str] = []

        mixture_mono = np.asarray(mixture, dtype=np.float32)
        if mixture_mono.ndim != 1 or len(mixture_mono) == 0:
            raise Track2PipelineError("Mixture must be a non-empty 1D mono NumPy array.")

        # 1. Determine enrollment audio
        enrollment_file_path = None
        speaker_label = selected_speaker or "Target Speaker"

        if enrollment_audio is not None:
            enroll_mono = np.asarray(enrollment_audio, dtype=np.float32)
            if enroll_mono.ndim != 1 or len(enroll_mono) == 0:
                raise Track2PipelineError("Enrollment audio must be a non-empty 1D mono array.")
        elif diarization_result is not None and selected_speaker is not None:
            # Build enrollment dynamically from diarized speaker segments
            enroll_res = self.enrollment_builder.build_enrollment(
                audio=mixture_mono,
                diarization_result=diarization_result,
                selected_speaker=selected_speaker,
            )
            enroll_mono = enroll_res.audio
            warnings.extend(enroll_res.warnings)
            enrollment_file_path = enroll_res.saved_path
        else:
            raise Track2PipelineError(
                "Either enrollment_audio OR (diarization_result and selected_speaker) must be provided."
            )

        # 2. Run PS4 Extraction
        extractor = self._ensure_model_loaded()
        extracted = extractor.extract(
            mixture=mixture_mono,
            enrollment=enroll_mono,
            sample_rate=16000,
        )

        # Refinement: Safeguard against energy collapse on quiet microphones
        ext_energy = float(np.mean(extracted ** 2))
        mix_energy = float(np.mean(mixture_mono ** 2))
        if ext_energy < 1e-7 and mix_energy > 1e-4:
            extracted = 0.5 * extracted + 0.5 * mixture_mono
            warnings.append("Applied dynamic gain compensation for low-energy extraction.")

        # 3. Output validation
        if not np.all(np.isfinite(extracted)):
            raise Track2PipelineError("Extracted audio contains non-finite values (NaN / Inf).")

        # Ensure extracted audio length matches mixture length
        if len(extracted) != len(mixture_mono):
            if len(extracted) < len(mixture_mono):
                extracted = np.pad(extracted, (0, len(mixture_mono) - len(extracted)))
            else:
                extracted = extracted[:len(mixture_mono)]

        # 4. Save to disk if requested
        saved_file = None
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), extracted, 16000, subtype="PCM_16")
            saved_file = str(out_p.resolve())

        # 5. Evaluate quality metrics ONLY when clean reference target is available
        metrics = None
        if clean_target is not None:
            target_mono = np.asarray(clean_target, dtype=np.float32)
            if target_mono.ndim == 1 and len(target_mono) == len(mixture_mono):
                try:
                    eval_sample = {
                        "mixture": mixture_mono,
                        "target": target_mono,
                        "sample_rate": 16000,
                    }
                    metrics = evaluate_sample(eval_sample, extracted)
                except Exception as eval_err:
                    warnings.append(f"Could not compute SI-SDR metrics: {eval_err}")
            else:
                warnings.append(
                    f"Clean target length ({len(target_mono)}) does not match mixture ({len(mixture_mono)}). "
                    f"Reference evaluation skipped."
                )

        exec_time = round(time.time() - start_time, 3)
        duration = round(float(len(extracted) / 16000), 3)

        return Track2Result(
            extracted_audio=extracted,
            mixture_audio=mixture_mono,
            enrollment_audio=enroll_mono,
            selected_speaker=speaker_label,
            sample_rate=16000,
            duration=duration,
            output_path=saved_file,
            enrollment_path=enrollment_file_path,
            metrics=metrics,
            warnings=warnings,
            execution_time=exec_time,
        )
