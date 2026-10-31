from pathlib import Path

import numpy as np
import soundfile as sf


# Locate the Echolens project folder
project_root = Path(__file__).resolve().parents[1]

# Input recordings
speaker_a_path = project_root / "data/raw/speaker_a_new_target.wav"
speaker_b_path = project_root / "data/raw/speaker_b.wav"

# Output folder
output_dir = project_root / "data/processed/training_samples"
output_dir.mkdir(parents=True, exist_ok=True)

# Configuration
sample_rate = 16000
segment_duration = 4
segment_samples = sample_rate * segment_duration
number_of_samples = 4

# Load recordings
speaker_a, sr_a = sf.read(speaker_a_path, dtype="float32")
speaker_b, sr_b = sf.read(speaker_b_path, dtype="float32")

# Validate recordings
if sr_a != sample_rate or sr_b != sample_rate:
    raise ValueError("Both recordings must have a sample rate of 16000 Hz.")

if speaker_a.ndim != 1 or speaker_b.ndim != 1:
    raise ValueError("Both recordings must be mono.")

if not np.all(np.isfinite(speaker_a)) or not np.all(np.isfinite(speaker_b)):
    raise ValueError("Recordings contain invalid audio values.")

required_samples = number_of_samples * segment_samples

if len(speaker_a) < required_samples or len(speaker_b) < required_samples:
    raise ValueError("Recordings are too short to create four samples.")

# Generate training samples
for index in range(number_of_samples):
    start = index * segment_samples
    end = start + segment_samples

    target = speaker_a[start:end].copy()
    interference = speaker_b[start:end].copy()

    # Mix the two speakers
    mixture = target + interference

    # Prevent clipping while preserving the target's relative volume
    peak = np.max(np.abs(mixture))
    scale = max(1.0, float(peak))

    mixture = mixture / scale
    target = target / scale

    sample_number = index + 1

    mixture_path = output_dir / f"mixture_{sample_number:02d}.wav"
    target_path = output_dir / f"target_{sample_number:02d}.wav"

    sf.write(mixture_path, mixture, sample_rate)
    sf.write(target_path, target, sample_rate)

    print(f"Sample {sample_number}: created successfully")

print("\nAll 4 training samples created!")
print(f"Output folder: {output_dir}")