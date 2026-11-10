import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Load audio recordings
clean, sr = sf.read("data/raw/speech_sample.wav")
noisy, sr_noisy = sf.read("data/processed/noisy_speech.wav")

# Validate audio
if sr != sr_noisy:
    raise ValueError("Sample rates do not match")

if clean.shape != noisy.shape:
    raise ValueError("Audio shapes do not match")

# Create time axis
time = np.arange(len(clean)) / sr

# Create comparison plots
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Clean speech
axes[0].plot(time, clean, color="blue")
axes[0].set_title("Echolens - Clean Speech")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Noisy speech
axes[1].plot(time, noisy, color="red")
axes[1].set_title("Echolens - Noisy Speech (10 dB SNR)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

# Added noise
noise = noisy - clean

axes[2].plot(time, noise, color="green")
axes[2].set_title("Echolens - Added Gaussian Noise")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

plt.tight_layout()
plt.show()