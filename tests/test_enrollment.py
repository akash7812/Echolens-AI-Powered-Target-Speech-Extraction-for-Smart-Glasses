"""
Tests for Target Speaker Enrollment Audio Builder
=================================================
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from echolens.diarization.base import (
    SpeakerSegment,
    SpeakerStats,
    DiarizationResult,
)
from echolens.enrollment import (
    EnrollmentBuilder,
    EnrollmentResult,
    EnrollmentError,
    build_enrollment,
)


@pytest.fixture
def multi_speaker_context():
    """Create synthetic audio and matching DiarizationResult for testing."""
    sr = 16000
    duration = 10.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.2

    # Speaker A: 3 segments totaling 4.5s
    # Speaker B: 1 segment of 1.2s (short, triggers warning)
    # Speaker C: 1 tiny segment of 0.3s (too short, triggers error)
    segments = [
        SpeakerSegment(speaker="Speaker A", start_time=0.5, end_time=2.5, duration=2.0, confidence=0.9),
        SpeakerSegment(speaker="Speaker B", start_time=2.6, end_time=3.8, duration=1.2, confidence=0.88),
        SpeakerSegment(speaker="Speaker A", start_time=4.0, end_time=5.5, duration=1.5, confidence=0.92),
        SpeakerSegment(speaker="Speaker C", start_time=6.0, end_time=6.3, duration=0.3, confidence=0.7),
        SpeakerSegment(speaker="Speaker A", start_time=7.0, end_time=8.0, duration=1.0, confidence=0.9),
    ]

    diar_result = DiarizationResult(
        segments=segments,
        speakers=["Speaker A", "Speaker B", "Speaker C"],
        speaker_stats={},
        total_audio_duration=duration,
        total_speech_duration=6.0,
        sample_rate=sr,
        num_speakers=3,
    )

    return audio, sr, diar_result


def test_build_enrollment_success(multi_speaker_context):
    """Check building standard enrollment for Speaker A."""
    audio, sr, diar_result = multi_speaker_context
    result = build_enrollment(audio, diar_result, selected_speaker="Speaker A")

    assert isinstance(result, EnrollmentResult)
    assert result.speaker == "Speaker A"
    assert result.sample_rate == 16000
    assert result.audio.ndim == 1
    assert result.duration == 4.5
    assert result.num_segments_used == 3
    assert result.rms_energy > 0.01
    assert len(result.warnings) == 0


def test_build_enrollment_caps_duration(multi_speaker_context):
    """Check that builder caps total enrollment duration to max_enrollment_duration."""
    audio, sr, diar_result = multi_speaker_context
    builder = EnrollmentBuilder(sample_rate=sr, max_enrollment_duration=3.0)
    result = builder.build_enrollment(audio, diar_result, selected_speaker="Speaker A")

    assert result.duration == 3.0
    assert len(result.audio) == int(3.0 * 16000)


def test_build_enrollment_saves_to_disk(tmp_path, multi_speaker_context):
    """Check saving canonical enrollment WAV file to disk."""
    audio, sr, diar_result = multi_speaker_context
    out_file = tmp_path / "enrollment_speaker_a.wav"

    result = build_enrollment(
        audio, diar_result, selected_speaker="Speaker A", save_path=out_file
    )

    assert out_file.is_file()
    assert result.saved_path == str(out_file.resolve())

    loaded, loaded_sr = sf.read(str(out_file))
    assert loaded_sr == 16000
    assert loaded.ndim == 1
    assert len(loaded) == len(result.audio)


def test_build_enrollment_short_warning(multi_speaker_context):
    """Check that speech duration between 0.5s and 1.5s produces a warning."""
    audio, sr, diar_result = multi_speaker_context
    result = build_enrollment(audio, diar_result, selected_speaker="Speaker B")

    assert result.speaker == "Speaker B"
    assert result.duration == 1.2
    assert len(result.warnings) >= 1
    assert any("relatively short" in w for w in result.warnings)


def test_build_enrollment_too_short_raises(multi_speaker_context):
    """Check that speech duration below 0.5s raises EnrollmentError."""
    audio, sr, diar_result = multi_speaker_context
    with pytest.raises(EnrollmentError, match="Insufficient speech for enrollment"):
        build_enrollment(audio, diar_result, selected_speaker="Speaker C")


def test_build_enrollment_unselected_speaker_raises(multi_speaker_context):
    """Check that selecting non-existent speaker raises EnrollmentError."""
    audio, sr, diar_result = multi_speaker_context
    with pytest.raises(EnrollmentError, match="not found in detected speakers"):
        build_enrollment(audio, diar_result, selected_speaker="Speaker Unknown")


def test_build_enrollment_silent_audio_raises():
    """Check that silent audio raises EnrollmentError."""
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    segments = [
        SpeakerSegment(speaker="Speaker A", start_time=0.5, end_time=2.5, duration=2.0, confidence=0.9),
    ]
    diar_result = DiarizationResult(
        segments=segments,
        speakers=["Speaker A"],
        speaker_stats={},
        total_audio_duration=4.0,
        total_speech_duration=2.0,
        sample_rate=sr,
        num_speakers=1,
    )

    with pytest.raises(EnrollmentError, match="near-silent"):
        build_enrollment(audio, diar_result, selected_speaker="Speaker A")


def test_build_enrollment_with_real_dataset():
    """Check end-to-end enrollment building using real dataset mixture."""
    mix_path = Path(r"C:\Echolens\data\processed\training_samples\mixture_01.wav")
    if mix_path.is_file():
        audio, sr = sf.read(str(mix_path), dtype="float32")
        from echolens.diarization import SpectralAHCDiarizer
        diarizer = SpectralAHCDiarizer(sample_rate=sr)
        diar_res = diarizer.diarize(audio, sample_rate=sr, num_speakers=2)

        # Choose the first speaker
        spk = diar_res.speakers[0]
        enroll_res = build_enrollment(audio, diar_res, selected_speaker=spk)

        assert enroll_res.speaker == spk
        assert enroll_res.sample_rate == 16000
        assert enroll_res.duration > 0.5
        assert np.all(np.isfinite(enroll_res.audio))
