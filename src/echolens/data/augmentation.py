import numpy as np


def add_gaussian_noise(audio, snr_db=10, seed=None):
    """
    Add Gaussian noise to audio at a specified SNR.

    Parameters:
        audio: NumPy array with shape (channels, samples)
        snr_db: Desired signal-to-noise ratio in decibels
        seed: Random seed for reproducibility

    Returns:
        Noisy audio with the same shape as the input
    """

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 2:
        raise ValueError("Audio must have shape (channels, samples)")

    if audio.size == 0:
        raise ValueError("Audio must not be empty")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio must contain only finite values")

    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")

    rng = np.random.default_rng(seed)

    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        return audio.copy()

    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=audio.shape
    ).astype(np.float32)

    noisy_audio = audio + noise

    return noisy_audio.astype(np.float32)