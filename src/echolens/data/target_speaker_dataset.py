from pathlib import Path

import numpy as np
import soundfile as sf


class TargetSpeakerDataset:
    def __init__(self, project_root, use_training_samples=False):
        self.project_root = Path(project_root)

        self.enrollment_path = (
            self.project_root / "data/raw/speaker_a_new_enrollment.wav"
        )

        if use_training_samples:
            # Load all generated 4-second samples
            samples_dir = (
                self.project_root / "data/processed/training_samples"
            )

            mixture_paths = sorted(samples_dir.glob("mixture_*.wav"))

            self.samples = []

            for mixture_path in mixture_paths:
                target_name = mixture_path.name.replace(
                    "mixture_", "target_", 1
                )
                target_path = samples_dir / target_name

                self.samples.append((mixture_path, target_path))

            if not self.samples:
                raise FileNotFoundError(
                    f"No training samples found in {samples_dir}"
                )

        else:
            # Preserve the original single-example behavior
            mixture_path = (
                self.project_root
                / "data/processed/speaker_mixture_new.wav"
            )

            target_path = (
                self.project_root
                / "data/processed/speaker_a_target_new.wav"
            )

            self.samples = [(mixture_path, target_path)]

        # Preserve original path attributes for compatibility
        self.mixture_path, self.target_path = self.samples[0]

        # Verify all required files exist
        required_paths = [self.enrollment_path]

        for mixture_path, target_path in self.samples:
            required_paths.extend([mixture_path, target_path])

        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found: {path}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset index must be an integer.")

        if index < 0 or index >= len(self.samples):
            raise IndexError("Dataset index is out of range.")

        mixture_path, target_path = self.samples[index]

        # Load audio
        mixture, sr_mix = sf.read(
            mixture_path, dtype="float32"
        )

        enrollment, sr_enroll = sf.read(
            self.enrollment_path, dtype="float32"
        )

        target, sr_target = sf.read(
            target_path, dtype="float32"
        )

        # Validate sample rates
        if not (sr_mix == sr_enroll == sr_target == 16000):
            raise ValueError(
                "All audio files must use 16000 Hz."
            )

        # Validate mono audio
        if any(
            audio.ndim != 1
            for audio in [mixture, enrollment, target]
        ):
            raise ValueError(
                "All audio files must be mono."
            )

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
                raise ValueError(
                    f"{name} audio is empty."
                )

            if not np.all(np.isfinite(audio)):
                raise ValueError(
                    f"{name} contains invalid values."
                )

        return {
            "mixture": mixture,
            "enrollment": enrollment,
            "target": target,
            "sample_rate": sr_mix,
        }