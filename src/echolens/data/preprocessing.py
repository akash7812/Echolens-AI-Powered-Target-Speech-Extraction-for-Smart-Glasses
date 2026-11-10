import numpy as np
from scipy.signal import resample_poly
from math import gcd


def resample_audio(audio, original_sr, target_sr=16000):
    """
    Resample multichannel audio.

    Expected audio shape: (channels, samples)
    """

    if original_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")

    if original_sr == target_sr:
        return audio.copy()

    divisor = gcd(original_sr, target_sr)

    up = target_sr // divisor
    down = original_sr // divisor

    resampled = resample_poly(
        audio,
        up,
        down,
        axis=-1
    )

    return resampled.astype(np.float32)


def normalize_audio(audio):
    """
    Peak-normalize audio without mixing channels.

    Note: Use only when appropriate for the model.
    """

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio.copy()

    return (audio / peak).astype(np.float32)


def validate_audio(audio, sample_rate):
    """
    Check audio shape, sample rate, and sample values.
    """

    if not isinstance(audio, np.ndarray):
        raise TypeError("Audio must be a NumPy array.")

    if audio.ndim != 2:
        raise ValueError(
            "Audio must have shape (channels, samples)."
        )

    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Audio cannot be empty.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinity.")

    return True