import numpy as np
import soundfile as sf

# Load clean and noisy speech
clean, sr_clean = sf.read(
    "data/raw/speech_sample.wav",
    dtype="float32"
)

noisy, sr_noisy = sf.read(
    "data/processed/noisy_speech.wav",
    dtype="float32"
)

# Verify sample rates and shapes
if sr_clean != sr_noisy:
    raise ValueError("Sample rates do not match")

if clean.shape != noisy.shape:
    raise ValueError("Audio shapes do not match")

# Calculate added noise
noise = noisy - clean

# Calculate signal and noise power
signal_power = np.mean(clean.astype(np.float64) ** 2)
noise_power = np.mean(noise.astype(np.float64) ** 2)

# Calculate SNR
if noise_power == 0:
    snr_db = float("inf")
elif signal_power == 0:
    snr_db = float("-inf")
else:
    snr_db = 10 * np.log10(signal_power / noise_power)

# Display results
print("Echolens - SNR Measurement")
print("-------------------------")
print("Sample Rate:", sr_clean)
print("Duration:", round(len(clean) / sr_clean, 2), "seconds")
print("Signal Power:", signal_power)
print("Noise Power:", noise_power)
print("Measured SNR:", round(snr_db, 2), "dB")