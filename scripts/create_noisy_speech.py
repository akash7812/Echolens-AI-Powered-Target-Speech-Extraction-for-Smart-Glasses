import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow Python to find our echolens package
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from echolens.data.augmentation import add_gaussian_noise

# Input and output paths
input_path = project_root / "data/raw/speech_sample.wav"
output_path = project_root / "data/processed/noisy_speech.wav"

# Load clean speech
audio, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)

# Convert (samples, channels) to (channels, samples)
audio = audio.T

# Add Gaussian noise at 10 dB SNR
noisy_audio = add_gaussian_noise(audio, snr_db=10, seed=42)

# Create output folder if needed
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save noisy speech
sf.write(output_path, noisy_audio.T, sample_rate, subtype="PCM_16")

# Display information
print("Noisy speech generated successfully!")
print("Sample Rate:", sample_rate)
print("Audio Shape:", noisy_audio.shape)
print("Duration:", round(noisy_audio.shape[1] / sample_rate, 2), "seconds")
print("Saved to:", output_path)