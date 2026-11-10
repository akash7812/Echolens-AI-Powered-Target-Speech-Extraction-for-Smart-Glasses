from pathlib import Path

import numpy as np
import soundfile as sf


# Project folder
project_root = Path(__file__).resolve().parents[1]

# Input recordings
speaker_a_path = project_root / "data/raw/speaker_a_new_target.wav"
speaker_b_path = project_root / "data/raw/speaker_b.wav"

# Output recordings
output_dir = project_root / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

mixture_path = output_dir / "speaker_mixture_new.wav"
target_path = output_dir / "speaker_a_target_new.wav"


# Load both recordings
speaker_a, sr_a = sf.read(speaker_a_path, dtype="float32")
speaker_b, sr_b = sf.read(speaker_b_path, dtype="float32")

# Check sample rates
if sr_a != sr_b:
    raise ValueError("Both recordings must have the same sample rate.")

# Check mono audio
if speaker_a.ndim != 1 or speaker_b.ndim != 1:
    raise ValueError("Both recordings must be mono.")

# Use the shorter recording length
min_length = min(len(speaker_a), len(speaker_b))

speaker_a = speaker_a[:min_length]
speaker_b = speaker_b[:min_length]

# Mix both speakers with equal gain
mixture = speaker_a + speaker_b

# Scale mixture and target together to prevent clipping
peak = np.max(np.abs(mixture))
scale = max(1.0, float(peak))

mixture = mixture / scale
target = speaker_a / scale

# Save recordings
sf.write(mixture_path, mixture, sr_a)
sf.write(target_path, target, sr_a)

print("Two-speaker mixture created successfully!")
print(f"Sample Rate: {sr_a}")
print(f"Duration: {len(mixture) / sr_a:.2f} seconds")
print(f"Mixture saved to: {mixture_path}")
print(f"Target saved to: {target_path}")