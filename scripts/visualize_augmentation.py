from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.audio_loader import load_audio
from echolens.data.augmentation import add_gaussian_noise


# Load original audio
audio_path = ROOT / "data" / "raw" / "sample.wav"
audio, sample_rate = load_audio(audio_path)

# Add background noise
noisy_audio = add_gaussian_noise(
    audio,
    snr_db=10,
    seed=42
)

# Select microphone channel 1
original = audio[0]
noisy = noisy_audio[0]

# Display first 20 milliseconds
duration = 0.02
samples = int(sample_rate * duration)

time = np.arange(samples) / sample_rate

# Create comparison plots
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(time, original[:samples])
axes[0].set_title("Original Audio - Microphone 1")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(time, noisy[:samples])
axes[1].set_title("Noisy Audio - Microphone 1 (10 dB SNR)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

plt.tight_layout()
plt.show()