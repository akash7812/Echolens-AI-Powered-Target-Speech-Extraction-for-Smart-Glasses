"""
Tests for Audio Ingestion and Format Standardization Module
============================================================
"""

from pathlib import Path
import io

import numpy as np
import pytest
import soundfile as sf

from echolens.audio.ingestion import (
    AudioIngestion,
    AudioIngestionError,
    IngestionResult,
    ingest_audio,
    TARGET_SAMPLE_RATE,
)


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Creates temporary directory with synthetic test audio files."""
    audio_dir = tmp_path / "audio_test"
    audio_dir.mkdir(parents=True)

    # 1. Standard 16 kHz mono (2 seconds)
    sr16 = 16000
    t16 = np.linspace(0, 2.0, sr16 * 2, endpoint=False)
    mono_16k = (0.5 * np.sin(2 * np.pi * 440 * t16)).astype(np.float32)
    sf.write(str(audio_dir / "mono_16k.wav"), mono_16k, sr16)

    # 2. Stereo 44.1 kHz (1 second)
    sr44 = 44100
    t44 = np.linspace(0, 1.0, sr44, endpoint=False)
    left = (0.4 * np.sin(2 * np.pi * 440 * t44)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 880 * t44)).astype(np.float32)
    stereo_44k = np.stack([left, right], axis=1)
    sf.write(str(audio_dir / "stereo_44k.wav"), stereo_44k, sr44)

    # 3. Very short audio (0.2 seconds - below min duration)
    short_audio = np.zeros(int(16000 * 0.2), dtype=np.float32)
    sf.write(str(audio_dir / "too_short.wav"), short_audio, 16000)

    return audio_dir


def test_ingest_standard_mono_wav(temp_audio_dir):
    """Check loading standard 16 kHz mono WAV file."""
    path = temp_audio_dir / "mono_16k.wav"
    result = ingest_audio(path)

    assert isinstance(result, IngestionResult)
    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.original_sample_rate == 16000
    assert result.original_channels == 1
    assert result.audio.ndim == 1
    assert len(result.audio) == 32000
    assert np.isclose(result.duration, 2.0)
    assert result.peak_amplitude > 0.4
    assert len(result.warnings) == 0


def test_ingest_resample_and_downmix(temp_audio_dir):
    """Check downmixing stereo to mono and resampling 44.1 kHz to 16 kHz."""
    path = temp_audio_dir / "stereo_44k.wav"
    result = ingest_audio(path)

    assert result.sample_rate == 16000
    assert result.original_sample_rate == 44100
    assert result.original_channels == 2
    assert result.audio.ndim == 1
    assert len(result.audio) == 16000  # 1.0 second at 16 kHz
    assert np.isclose(result.duration, 1.0)
    assert any("Downmixed 2 channels" in w for w in result.warnings)
    assert any("Resampled audio from 44100 Hz" in w for w in result.warnings)


def test_ingest_from_numpy_array():
    """Check ingesting raw numpy array with explicit sample rate."""
    sr = 48000
    samples = int(sr * 1.5)
    t = np.linspace(0, 1.5, samples, endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    ingestion = AudioIngestion()
    result = ingestion.ingest(signal, input_sample_rate=sr)

    assert result.sample_rate == 16000
    assert result.num_samples == int(16000 * 1.5)
    assert np.isclose(result.duration, 1.5)
    assert result.audio.ndim == 1


def test_ingest_from_bytes():
    """Check ingesting audio bytes from in-memory WAV buffer."""
    sr = 16000
    audio_data = (0.2 * np.random.randn(16000 * 2)).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format="WAV")
    raw_bytes = buffer.getvalue()

    result = ingest_audio(raw_bytes)
    assert result.sample_rate == 16000
    assert result.num_samples == 32000
    assert np.isclose(result.duration, 2.0)


def test_ingest_saves_to_disk(temp_audio_dir):
    """Check saving canonical 16 kHz mono audio file to disk."""
    path = temp_audio_dir / "stereo_44k.wav"
    out_path = temp_audio_dir / "saved_canonical.wav"

    result = ingest_audio(path, save_path=out_path)

    assert out_path.is_file()
    assert result.saved_path == str(out_path.resolve())

    # Verify saved file on disk
    loaded_data, loaded_sr = sf.read(str(out_path))
    assert loaded_sr == 16000
    assert loaded_data.ndim == 1
    assert len(loaded_data) == 16000


def test_ingest_too_short_raises(temp_audio_dir):
    """Check that audio shorter than MIN_DURATION raises AudioIngestionError."""
    path = temp_audio_dir / "too_short.wav"
    with pytest.raises(AudioIngestionError, match="shorter than minimum allowed"):
        ingest_audio(path)


def test_ingest_missing_file_raises():
    """Check that missing file path raises AudioIngestionError."""
    with pytest.raises(AudioIngestionError, match="not found"):
        ingest_audio(r"C:\non_existent_audio_path_xyz.wav")


def test_ingest_nan_values_raises():
    """Check that non-finite audio values raise AudioIngestionError."""
    corrupt_audio = np.array([0.1, np.nan, 0.5, 0.2], dtype=np.float32)
    with pytest.raises(AudioIngestionError, match="non-finite"):
        ingest_audio(corrupt_audio, input_sample_rate=16000)


def test_ingest_silence_warning():
    """Check that near-silent audio produces a warning rather than crashing."""
    silent_audio = np.zeros(16000 * 2, dtype=np.float32)
    result = ingest_audio(silent_audio, input_sample_rate=16000)

    assert result.sample_rate == 16000
    assert any("near-silent" in w for w in result.warnings)


def test_ingest_clipping_warning():
    """Check that audio reaching +/-1.0 produces clipping warning."""
    clipped_audio = np.array([1.0] * 16000, dtype=np.float32)
    result = ingest_audio(clipped_audio, input_sample_rate=16000)

    assert any("clipping" in w for w in result.warnings)


def test_ingest_real_mixture_file():
    """Check ingesting real project mixture file from data/processed."""
    mixture_path = Path(r"C:\Echolens\data\processed\training_samples\mixture_01.wav")
    if mixture_path.is_file():
        result = ingest_audio(mixture_path)
        assert result.sample_rate == 16000
        assert result.num_samples == 64000  # 4 seconds
        assert np.isclose(result.duration, 4.0)
        assert result.audio.ndim == 1
