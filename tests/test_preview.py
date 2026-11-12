"""
Tests for Speaker Preview and Manual Selection Module
=====================================================
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
from echolens.diarization.preview import (
    SpeakerPreview,
    SpeakerPreviewGenerator,
    SpeakerSelectionManager,
    SpeakerSelectionError,
    MIN_ENROLLMENT_DURATION_WARN,
    MIN_ENROLLMENT_DURATION_ERROR,
)


@pytest.fixture
def sample_diarization_data():
    """Create synthetic audio and matching DiarizationResult with 2 speakers."""
    sr = 16000
    duration = 6.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    # Speaker A: 2 segments totaling 2.5 seconds
    # Speaker B: 1 segment of 1.0 second (insufficient for enrollment warning)
    segments = [
        SpeakerSegment(speaker="Speaker A", start_time=0.5, end_time=2.0, duration=1.5, confidence=0.9),
        SpeakerSegment(speaker="Speaker B", start_time=2.2, end_time=3.2, duration=1.0, confidence=0.85),
        SpeakerSegment(speaker="Speaker A", start_time=3.5, end_time=4.5, duration=1.0, confidence=0.9),
    ]

    stats = {
        "Speaker A": SpeakerStats(speaker="Speaker A", total_duration=2.5, num_segments=2, percentage=41.7, avg_segment_duration=1.25),
        "Speaker B": SpeakerStats(speaker="Speaker B", total_duration=1.0, num_segments=1, percentage=16.7, avg_segment_duration=1.0),
    }

    result = DiarizationResult(
        segments=segments,
        speakers=["Speaker A", "Speaker B"],
        speaker_stats=stats,
        total_audio_duration=duration,
        total_speech_duration=3.5,
        sample_rate=sr,
        num_speakers=2,
    )

    return audio, sr, result


def test_preview_generation(sample_diarization_data):
    """Check preview generator produces preview audio for every detected speaker."""
    audio, sr, diar_result = sample_diarization_data
    generator = SpeakerPreviewGenerator(sample_rate=sr, target_preview_duration=2.0)
    previews = generator.generate_previews(audio, diar_result)

    assert len(previews) == 2
    assert "Speaker A" in previews
    assert "Speaker B" in previews

    prev_a = previews["Speaker A"]
    assert isinstance(prev_a, SpeakerPreview)
    assert prev_a.speaker == "Speaker A"
    assert prev_a.sample_rate == sr
    assert prev_a.duration > 1.0
    assert prev_a.total_detected_duration == 2.5
    assert prev_a.is_sufficient_for_enrollment is True
    assert prev_a.quality_warning is None

    prev_b = previews["Speaker B"]
    assert prev_b.speaker == "Speaker B"
    assert prev_b.total_detected_duration == 1.0
    # Speaker B only has 1.0s, which is below 1.5s threshold
    assert prev_b.is_sufficient_for_enrollment is False
    assert prev_b.quality_warning is not None
    assert "Warning" in prev_b.quality_warning


def test_preview_file_saving(tmp_path, sample_diarization_data):
    """Check saving preview clips to disk as standard WAV files."""
    audio, sr, diar_result = sample_diarization_data
    out_dir = tmp_path / "previews"
    generator = SpeakerPreviewGenerator(sample_rate=sr)
    previews = generator.generate_previews(audio, diar_result, output_dir=out_dir)

    for spk, prev in previews.items():
        assert prev.saved_path is not None
        p = Path(prev.saved_path)
        assert p.is_file()
        loaded, loaded_sr = sf.read(str(p))
        assert loaded_sr == sr
        assert loaded.ndim == 1


def test_preview_critical_short_speech():
    """Check that speaker with < 0.5s gets critical warning and cannot be enrolled."""
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)
    segments = [
        SpeakerSegment(speaker="Speaker A", start_time=0.1, end_time=0.4, duration=0.3, confidence=0.8),
    ]
    diar_result = DiarizationResult(
        segments=segments,
        speakers=["Speaker A"],
        speaker_stats={},
        total_audio_duration=2.0,
        total_speech_duration=0.3,
        sample_rate=sr,
        num_speakers=1,
    )
    generator = SpeakerPreviewGenerator(sample_rate=sr)
    previews = generator.generate_previews(audio, diar_result)

    prev_a = previews["Speaker A"]
    assert prev_a.is_sufficient_for_enrollment is False
    assert prev_a.quality_warning is not None
    assert "Critical" in prev_a.quality_warning


def test_selection_manager_no_default_assumption():
    """Verify that the selection manager NEVER assumes Speaker A is the target by default."""
    manager = SpeakerSelectionManager(["Speaker A", "Speaker B", "Speaker C"])

    assert manager.selected_speaker is None
    assert manager.is_confirmed is False

    # Attempting to confirm without explicit selection must fail
    with pytest.raises(SpeakerSelectionError, match="No speaker has been selected"):
        manager.confirm()


def test_selection_manager_explicit_selection_flow():
    """Check selecting a speaker, checking status, and explicitly confirming."""
    manager = SpeakerSelectionManager(["Speaker A", "Speaker B"])

    # User explicitly chooses Speaker B
    manager.select("Speaker B")
    assert manager.selected_speaker == "Speaker B"
    assert manager.is_confirmed is False

    # Confirm
    confirmed = manager.confirm()
    assert confirmed == "Speaker B"
    assert manager.is_confirmed is True


def test_selection_manager_switch_and_reconfirm():
    """Check changing selection requires re-confirmation."""
    manager = SpeakerSelectionManager(["Speaker A", "Speaker B"])

    manager.select("Speaker A")
    manager.confirm()
    assert manager.is_confirmed is True

    # User changes mind and selects Speaker B
    manager.select("Speaker B")
    assert manager.selected_speaker == "Speaker B"
    assert manager.is_confirmed is False  # Must re-confirm!

    manager.confirm()
    assert manager.selected_speaker == "Speaker B"
    assert manager.is_confirmed is True


def test_selection_manager_invalid_speaker_raises():
    """Check selecting a non-existent speaker raises error."""
    manager = SpeakerSelectionManager(["Speaker A", "Speaker B"])

    with pytest.raises(SpeakerSelectionError, match="not in detected list"):
        manager.select("Speaker Z")


def test_selection_manager_reset():
    """Check reset clears both selection and confirmation."""
    manager = SpeakerSelectionManager(["Speaker A", "Speaker B"])
    manager.select("Speaker A")
    manager.confirm()

    manager.reset()
    assert manager.selected_speaker is None
    assert manager.is_confirmed is False


def test_selection_manager_empty_speakers_raises():
    """Check selection on empty speaker list raises error."""
    manager = SpeakerSelectionManager([])

    with pytest.raises(SpeakerSelectionError, match="No speakers have been detected"):
        manager.select("Speaker A")
