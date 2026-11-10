
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .inference import build_model, load_checkpoint, extract_speaker


class PS4Extractor:
    """Echolens wrapper for the pretrained PS4 target speech extraction model."""

    SAMPLE_RATE = 16000

    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PS4 checkpoint not found: {self.checkpoint_path}"
            )

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        print(f"[Echolens] Initializing PS4 on {self.device}...")

        self.model = build_model(self.device)
        load_checkpoint(
            str(self.checkpoint_path),
            self.model,
            self.device
        )
        self.model.eval()

        print("[Echolens] PS4 model is ready.")

    def _prepare_audio(self, audio, sample_rate):
        """Convert mono NumPy audio into a 16 kHz PyTorch tensor."""

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError("Audio must be mono (one-dimensional).")

        if audio.size == 0:
            raise ValueError("Audio cannot be empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or infinite values.")

        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sample_rate != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE
            )

        return waveform

    def extract(self, mixture, enrollment, sample_rate=16000):
        """
        Extract the enrolled target speaker from a mixture.

        Args:
            mixture: Mono NumPy audio array.
            enrollment: Mono NumPy enrollment audio array.
            sample_rate: Input sample rate in Hz.

        Returns:
            Extracted target speech as a NumPy float32 array.
        """

        mixture_tensor = self._prepare_audio(mixture, sample_rate)
        enrollment_tensor = self._prepare_audio(enrollment, sample_rate)

        extracted = extract_speaker(
            self.model,
            mixture_tensor,
            enrollment_tensor,
            self.device
        )

        return extracted.squeeze(0).numpy().astype(np.float32)
