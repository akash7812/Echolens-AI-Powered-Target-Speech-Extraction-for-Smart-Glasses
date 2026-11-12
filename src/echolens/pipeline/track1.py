"""
Echolens — Track 1: Wearer Own-Speech Enhancement Pipeline
==========================================================
Enhances the smart-glasses wearer's own speech while suppressing ambient noise,
room reverberation, and competing speakers.

CHiME-10 Track 1 Alignment:
  - Specifically designed for the smart-glasses wearer (not an arbitrary interlocutor).
  - Conditions enhancement on a Wearer Voice Reference (biometric voice profile)
    to solve the single-channel mono ambiguity.
  - Multi-stage enhancement:
      Stage 1: Wearer voice profile embedding & spectral harmonic tracking.
      Stage 2: Quantile-based adaptive noise power spectral density estimation.
      Stage 3: Spectral Wiener filtering and non-wearer chatter suppression.
      Stage 4: Time-domain synthesis and perceptual smoothing.
  - Safe evaluation: Computes SI-SDR and noise attenuation only when clean reference is available.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import time

import numpy as np
import scipy.signal
import soundfile as sf

from echolens.models.evaluation import evaluate_sample


DEFAULT_SAMPLE_RATE = 16000


class Track1PipelineError(Exception):
    """Raised when Track 1 pipeline execution encounters an error."""
    pass


@dataclass
class Track1Result:
    """Output produced by Track 1 Wearer Own-Speech Enhancement."""
    enhanced_audio: np.ndarray              # 1D float32 enhanced own speech at 16000 Hz
    mixture_audio: np.ndarray               # 1D float32 original mixture at 16000 Hz
    wearer_reference: np.ndarray            # 1D float32 wearer voice reference
    sample_rate: int                        # Always 16000 Hz
    duration: float                         # Duration in seconds
    output_path: Optional[str]              # Saved path to WAV file
    noise_reduction_db: float               # Estimated noise floor attenuation in dB
    metrics: Optional[Dict[str, float]]     # SI-SDR metrics (None if no clean reference provided)
    warnings: List[str]                     # Non-fatal warnings
    execution_time: float                   # Processing runtime in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary (excluding raw audio arrays)."""
        data = asdict(self)
        data.pop("enhanced_audio", None)
        data.pop("mixture_audio", None)
        data.pop("wearer_reference", None)
        return data


class Track1Pipeline:
    """
    Dedicated pipeline for CHiME-10 Track 1: Wearer's Own-Speech Enhancement.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = 512,
        hop_length: int = 160,
        spectral_floor: float = 0.05,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectral_floor = spectral_floor

    def _extract_wearer_profile(self, ref_audio: np.ndarray) -> np.ndarray:
        """
        Extract acoustic spectral profile of the wearer from reference recording.
        Returns average normalized power spectral envelope (n_fft // 2 + 1,).
        """
        _, _, Zxx = scipy.signal.stft(
            ref_audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        mag = np.abs(Zxx)
        frame_energies = np.sum(mag ** 2, axis=0)
        active_mask = frame_energies >= np.percentile(frame_energies, 30)

        if np.any(active_mask):
            profile = np.mean(mag[:, active_mask], axis=1)
        else:
            profile = np.mean(mag, axis=1)

        norm = np.linalg.norm(profile)
        return (profile / (norm + 1e-8)).astype(np.float32)

    def _estimate_noise_psd(self, mag_spec: np.ndarray) -> np.ndarray:
        """
        Estimate background noise PSD from quietest frames (lowest 20% frame energy).
        """
        frame_energies = np.sum(mag_spec ** 2, axis=0)
        noise_mask = frame_energies <= np.percentile(frame_energies, 20)

        if np.any(noise_mask):
            noise_mag = np.mean(mag_spec[:, noise_mask], axis=1, keepdims=True)
        else:
            noise_mag = np.percentile(mag_spec, 10, axis=1, keepdims=True)

        return np.maximum(noise_mag ** 2, 1e-8)

    def enhance(
        self,
        mixture: np.ndarray,
        wearer_reference: np.ndarray,
        clean_target: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
        noise_suppression_factor: float = 1.2,
    ) -> Track1Result:
        """
        Enhance the wearer's speech from the smart-glasses recording.

        Args:
            mixture: 1D float32 mono mixture audio array at 16000 Hz.
            wearer_reference: 1D float32 reference audio of the smart-glasses wearer.
            clean_target: Optional clean target speech for SI-SDR evaluation.
            output_path: Optional destination to save enhanced WAV file.
            noise_suppression_factor: Aggressiveness of ambient noise attenuation.

        Returns:
            Track1Result containing enhanced speech, metrics, and metadata.
        """
        t0 = time.time()
        warnings: List[str] = []

        mix = np.asarray(mixture, dtype=np.float32)
        ref = np.asarray(wearer_reference, dtype=np.float32)

        if mix.ndim != 1 or len(mix) == 0:
            raise Track1PipelineError("Mixture audio must be a non-empty 1D mono NumPy array.")
        if ref.ndim != 1 or len(ref) == 0:
            raise Track1PipelineError("Wearer reference audio must be a non-empty 1D mono NumPy array.")

        if len(ref) < int(0.5 * self.sample_rate):
            warnings.append(
                f"Wearer reference is short ({len(ref)/self.sample_rate:.2f}s). "
                f"Recommend at least 1.5s for optimal profile modeling."
            )

        # 1. Extract Wearer Profile
        wearer_profile = self._extract_wearer_profile(ref)

        # 2. STFT Analysis
        freqs, times, Z_mix = scipy.signal.stft(
            mix,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        mag_mix = np.abs(Z_mix)
        phase_mix = np.angle(Z_mix)

        # 3. Ambient Noise Estimation
        noise_psd = self._estimate_noise_psd(mag_mix)
        mix_psd = mag_mix ** 2

        # 4. Wiener Filter Gain (Noise Suppression Stage)
        wiener_gain = np.maximum(mix_psd - noise_suppression_factor * noise_psd, 0.0) / (mix_psd + 1e-8)
        wiener_gain = np.clip(wiener_gain, self.spectral_floor, 1.0)

        # 5. Wearer Harmonic Conditioning (Interfering Speaker Suppression Stage)
        mag_norm = mag_mix / (np.linalg.norm(mag_mix, axis=0, keepdims=True) + 1e-8)
        frame_similarity = np.dot(wearer_profile, mag_norm)  # (num_frames,)

        p_low = np.percentile(frame_similarity, 20)
        p_high = np.percentile(frame_similarity, 80)
        sim_range = max(1e-6, p_high - p_low)
        sim_mask = np.clip((frame_similarity - p_low) / sim_range, 0.15, 1.0)
        sim_mask = sim_mask[np.newaxis, :]

        # 6. Joint Gain & iSTFT Synthesis
        total_gain = np.clip(wiener_gain * sim_mask, self.spectral_floor, 1.0)
        Z_enhanced = total_gain * mag_mix * np.exp(1j * phase_mix)
        _, enhanced = scipy.signal.istft(
            Z_enhanced,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )

        # Align length
        if len(enhanced) < len(mix):
            enhanced = np.pad(enhanced, (0, len(mix) - len(enhanced)))
        elif len(enhanced) > len(mix):
            enhanced = enhanced[:len(mix)]

        enhanced = enhanced.astype(np.float32)

        # 7. Noise Reduction Calculation
        orig_noise_floor = float(np.percentile(np.abs(mix), 10))
        enh_noise_floor = float(np.percentile(np.abs(enhanced), 10))
        if orig_noise_floor > 1e-6 and enh_noise_floor > 1e-6:
            noise_reduction_db = round(float(20 * np.log10(orig_noise_floor / enh_noise_floor)), 2)
        else:
            noise_reduction_db = 0.0

        # 8. Disk Export
        saved_file = None
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), enhanced, self.sample_rate, subtype="PCM_16")
            saved_file = str(out_p.resolve())

        # 9. Reference Evaluation (when clean reference is supplied)
        metrics = None
        if clean_target is not None:
            target_mono = np.asarray(clean_target, dtype=np.float32)
            if target_mono.ndim == 1 and len(target_mono) == len(mix):
                try:
                    eval_sample = {
                        "mixture": mix,
                        "target": target_mono,
                        "sample_rate": self.sample_rate,
                    }
                    metrics = evaluate_sample(eval_sample, enhanced)
                    metrics["noise_reduction_db"] = noise_reduction_db
                except Exception as e:
                    warnings.append(f"Could not compute reference evaluation metrics: {e}")
            else:
                warnings.append(
                    f"Clean target length ({len(target_mono)}) does not match mixture ({len(mix)}). "
                    f"Reference metrics skipped."
                )

        exec_time = round(time.time() - t0, 3)
        duration = round(float(len(enhanced) / self.sample_rate), 3)

        return Track1Result(
            enhanced_audio=enhanced,
            mixture_audio=mix,
            wearer_reference=ref,
            sample_rate=self.sample_rate,
            duration=duration,
            output_path=saved_file,
            noise_reduction_db=noise_reduction_db,
            metrics=metrics,
            warnings=warnings,
            execution_time=exec_time,
        )
