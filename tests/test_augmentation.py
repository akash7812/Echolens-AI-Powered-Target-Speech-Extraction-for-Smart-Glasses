import numpy as np

from echolens.data.augmentation import add_gaussian_noise


def test_noise_preserves_shape():
    audio = np.ones((2, 16000), dtype=np.float32)

    noisy = add_gaussian_noise(audio, snr_db=10, seed=42)

    assert noisy.shape == audio.shape
    assert noisy.dtype == np.float32


def test_noise_is_reproducible():
    audio = np.ones((2, 16000), dtype=np.float32)

    noisy_1 = add_gaussian_noise(audio, snr_db=10, seed=42)
    noisy_2 = add_gaussian_noise(audio, snr_db=10, seed=42)

    np.testing.assert_array_equal(noisy_1, noisy_2)


def test_silent_audio():
    audio = np.zeros((2, 16000), dtype=np.float32)

    noisy = add_gaussian_noise(audio, snr_db=10, seed=42)

    np.testing.assert_array_equal(noisy, audio)


def test_noise_snr():
    audio = np.ones((2, 160000), dtype=np.float32)

    noisy = add_gaussian_noise(audio, snr_db=10, seed=42)

    noise = noisy - audio

    signal_power = np.mean(audio.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)

    actual_snr = 10 * np.log10(signal_power / noise_power)

    assert abs(actual_snr - 10) < 0.2