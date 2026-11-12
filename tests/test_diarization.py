"""
Tests for Speaker Diarization Module
====================================
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from echolens.diarization import (
    SpectralAHCDiarizer,
    DiarizationResult,
    SpeakerSegment,
)


@pytest.fixture
def synthetic_conversational_audio():
    """
    Generate synthetic conversation alternating between two synthetic voices:
    Voice 1: 300 Hz fundamental + harmonics (0.0s - 1.5s, 3.0s - 4.5s)
    Voice 2: 700 Hz fundamental + harmonics (1.5s - 3.0s, 4.5s - 6.0s)
    """
    sr = 16000
    duration = 6.0
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Voice 1
    mask_v1 = ((t >= 0.0) & (t < 1.5)) | ((t >= 3.0) & (t < 4.5))
    v1 = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
    audio[mask_v1] = v1[mask_v1]

    # Voice 2
    mask_v2 = ((t >= 1.5) & (t < 3.0)) | ((t >= 4.5) & (t < 6.0))
    v2 = 0.4 * np.sin(2 * np.pi * 700 * t) + 0.2 * np.sin(2 * np.pi * 1400 * t)
    audio[mask_v2] = v2[mask_v2]

    return audio, sr


def test_diarization_synthetic_conversation(synthetic_conversational_audio):
    """Check diarizing multi-speaker conversation detects distinct speakers."""
    audio, sr = synthetic_conversational_audio
    diarizer = SpectralAHCDiarizer(sample_rate=sr)
    result = diarizer.diarize(audio, sample_rate=sr, num_speakers=2)

    assert isinstance(result, DiarizationResult)
    assert result.num_speakers == 2
    assert "Speaker A" in result.speakers
    assert "Speaker B" in result.speakers
    assert len(result.segments) >= 2

    # Check timestamps validity
    for seg in result.segments:
        assert isinstance(seg, SpeakerSegment)
        assert seg.speaker in ["Speaker A", "Speaker B"]
        assert seg.start_time >= 0.0
        assert seg.end_time <= 6.01
        assert seg.start_time < seg.end_time
        assert seg.duration > 0.0

    # Check stats
    stats_a = result.speaker_stats["Speaker A"]
    stats_b = result.speaker_stats["Speaker B"]
    assert stats_a.total_duration > 0.5
    assert stats_b.total_duration > 0.5


def test_diarization_silence():
    """Check that pure silence returns 0 speakers and empty segments without crashing."""
    sr = 16000
    silent_audio = np.zeros(sr * 3, dtype=np.float32)
    diarizer = SpectralAHCDiarizer(sample_rate=sr)
    result = diarizer.diarize(silent_audio, sample_rate=sr)

    assert result.num_speakers == 0
    assert len(result.speakers) == 0
    assert len(result.segments) == 0
    assert result.total_speech_duration == 0.0


def test_diarization_single_speaker():
    """Check that single-speaker audio is labeled with 1 speaker ('Speaker A')."""
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    single_voice = (0.4 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

    diarizer = SpectralAHCDiarizer(sample_rate=sr)
    result = diarizer.diarize(single_voice, sample_rate=sr, num_speakers=1)

    assert result.num_speakers == 1
    assert result.speakers == ["Speaker A"]
    assert len(result.segments) >= 1
    for seg in result.segments:
        assert seg.speaker == "Speaker A"


def test_diarization_invalid_sample_rate():
    """Check that passing wrong sample rate raises ValueError."""
    audio = np.zeros(16000, dtype=np.float32)
    diarizer = SpectralAHCDiarizer(sample_rate=16000)

    with pytest.raises(ValueError, match="Sample rate must be 16000"):
        diarizer.diarize(audio, sample_rate=44100)


def test_diarization_invalid_shape():
    """Check that non-mono audio raises ValueError."""
    audio_2d = np.zeros((2, 16000), dtype=np.float32)
    diarizer = SpectralAHCDiarizer(sample_rate=16000)

    with pytest.raises(ValueError, match="mono"):
        diarizer.diarize(audio_2d, sample_rate=16000)


def test_diarization_result_serialization(synthetic_conversational_audio):
    """Check that to_dict() returns JSON-serializable structure with all required fields."""
    audio, sr = synthetic_conversational_audio
    diarizer = SpectralAHCDiarizer(sample_rate=sr)
    result = diarizer.diarize(audio, sample_rate=sr, num_speakers=2)

    d = result.to_dict()
    assert "num_speakers" in d
    assert "speakers" in d
    assert "segments" in d
    assert "speaker_stats" in d
    assert isinstance(d["segments"], list)
    assert isinstance(d["speaker_stats"], dict)


def test_diarization_real_dataset_mixture():
    """Check diarizing real mixture recording from C:/Echolens/data."""
    mixture_path = Path(r"C:\Echolens\data\processed\training_samples\mixture_01.wav")
    if mixture_path.is_file():
        audio, sr = sf.read(str(mixture_path), dtype="float32")
        diarizer = SpectralAHCDiarizer(sample_rate=sr)
        result = diarizer.diarize(audio, sample_rate=sr, num_speakers=2)

        assert result.num_speakers == 2
        assert "Speaker A" in result.speakers
        assert "Speaker B" in result.speakers
        assert len(result.segments) >= 1
        assert result.total_speech_duration > 0.0
