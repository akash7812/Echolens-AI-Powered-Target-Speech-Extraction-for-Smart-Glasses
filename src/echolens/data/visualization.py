import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# FUNCTION 1: WAVEFORM VISUALIZATION
# ==========================================

def plot_waveform(audio, sample_rate):
    """
    Plot waveforms for all microphone channels.
    """

    channels, samples = audio.shape

    time = np.arange(samples) / sample_rate

    fig, axes = plt.subplots(
        channels,
        1,
        figsize=(12, 3 * channels),
        sharex=True,
        squeeze=False
    )

    for channel in range(channels):
        ax = axes[channel, 0]

        ax.plot(time, audio[channel])

        # Zoom into first 20 milliseconds
        ax.set_xlim(0, 0.02)

        ax.set_title(f"Microphone Channel {channel + 1}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    axes[-1, 0].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()


# ==========================================
# FUNCTION 2: SPECTROGRAM VISUALIZATION
# ==========================================

def plot_spectrogram(audio, sample_rate):
    """
    Display spectrograms for all microphone channels.
    """

    channels, samples = audio.shape

    fig, axes = plt.subplots(
        channels,
        1,
        figsize=(12, 4 * channels),
        squeeze=False
    )

    for channel in range(channels):
        ax = axes[channel, 0]

        spectrum = ax.specgram(
            audio[channel],
            NFFT=512,
            Fs=sample_rate,
            noverlap=256,
            cmap="viridis"
        )

        ax.set_title(
            f"Microphone Channel {channel + 1} - Spectrogram"
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")

        fig.colorbar(
            spectrum[3],
            ax=ax,
            label="Power (dB)"
        )

    plt.tight_layout()
    plt.show()