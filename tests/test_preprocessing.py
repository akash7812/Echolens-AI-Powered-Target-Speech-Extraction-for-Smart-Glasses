import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.preprocessing import (
    resample_audio,
    normalize_audio,
    validate_audio,
)


def test_resampling():
    audio = np.random.randn(2, 48000).astype(np.float32)

    result = resample_audio(audio, 48000, 16000)

    assert result.shape == (2, 16000)


def test_normalization():
    audio = np.array([[0.2, -0.4, 0.8]], dtype=np.float32)

    result = normalize_audio(audio)

    assert np.isclose(np.max(np.abs(result)), 1.0)


def test_validation():
    audio = np.zeros((2, 16000), dtype=np.float32)

    assert validate_audio(audio, 16000) is True