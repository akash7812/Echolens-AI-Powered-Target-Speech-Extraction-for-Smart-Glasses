from pathlib import Path

import numpy as np
import soundfile as sf


class TargetSpeakerDataset:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

        # Audio file paths
        self.mixture_path = (
            self.project_root / "data/processed/speaker_mixture_new.wav"
        )

        self.enrollment_path = (
            self.project_root / "data/raw/speaker_a_new_enrollment.wav"
        )

        self.target_path = (
            self.project_root / "data/processed/speaker_a_target_new.wav"
        )

        # Check that all files exist
        for path in [
            self.mixture_path,
            self.enrollment_path,
            self.target_path,
        ]:
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")

    def __len__(self):
        # Currently, we have one training example
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("Dataset currently contains only one example.")

        # Load audio
        mixture, sr_mix = sf.read(self.mixture_path, dtype="float32")
        enrollment, sr_enroll = sf.read(
            self.enrollment_path, dtype="float32"
        )
        target, sr_target = sf.read(self.target_path, dtype="float32")

        # Check sample rates
        if not (sr_mix == sr_enroll == sr_target == 16000):
            raise ValueError("All audio files must use 16000 Hz.")

        # Check mono audio
        if any(
            audio.ndim != 1
            for audio in [mixture, enrollment, target]
        ):
            raise ValueError("All audio files must be mono.")

        # Mixture and target must have equal lengths
        if len(mixture) != len(target):
            raise ValueError(
                "Mixture and target must have the same number of samples."
            )

        # Check for empty or invalid audio
        for name, audio in [
            ("mixture", mixture),
            ("enrollment", enrollment),
            ("target", target),
        ]:
            if len(audio) == 0:
                raise ValueError(f"{name} audio is empty.")

            if not np.all(np.isfinite(audio)):
                raise ValueError(f"{name} contains invalid values.")

        return {
            "mixture": mixture,
            "enrollment": enrollment,
            "target": target,
            "sample_rate": sr_mix,
        }