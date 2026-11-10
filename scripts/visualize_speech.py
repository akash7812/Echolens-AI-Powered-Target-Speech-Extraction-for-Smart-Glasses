import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

# Load speech recording
audio_path = "data/raw/speech_sample.wav"

audio, sample_rate = sf.read(audio_path)

# Create time axis
time = np.arange(len(audio)) / sample_rate

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot waveform
axes[0].plot(time, audio)
axes[0].set_title("Echolens - Speech Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Calculate spectrogram
spectrogram = librosa.stft(audio)

spectrogram_db = librosa.amplitude_to_db(
    np.abs(spectrogram),
    ref=np.max
)

# Plot spectrogram
img = librosa.display.specshow(
    spectrogram_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)

axes[1].set_title("Echolens - Speech Spectrogram")

fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()