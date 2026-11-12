"""
Tests for Track 2 Target Speaker Extraction Pipeline (PS4 Integration)
======================================================================
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from echolens.pipeline.track2 import (
    Track2Pipeline,
    Track2Result,
    Track2PipelineError,
    DEFAULT_CHECKPOINT_PATH,
)
from echolens.diarization.base import (
    SpeakerSegment,
    DiarizationResult,
)


@pytest.fixture
def real_checkpoint_available():
    """Verify that the PS4 model checkpoint exists."""
    p = Path(DEFAULT_CHECKPOINT_PATH)
    if not p.is_file():
        pytest.skip(f"PS4 checkpoint not found at {p}")
    return p


@pytest.fixture
def test_audio_data():
    """Load or generate short 1-second test audio arrays at 16 kHz."""
    sr = 16000
    mix_path = Path(r"C:\Echolens\data\processed\training_samples\mixture_01.wav")
    target_path = Path(r"C:\Echolens\data\processed\training_samples\target_01.wav")
    enroll_path = Path(r"C:\Echolens\data\raw\speaker_a_new_enrollment.wav")

    if mix_path.is_file() and target_path.is_file() and enroll_path.is_file():
        mix, _ = sf.read(str(mix_path), dtype="float32")
        target, _ = sf.read(str(target_path), dtype="float32")
        enroll, _ = sf.read(str(enroll_path), dtype="float32")
        # Use 1 second for fast test execution on CPU
        return mix[:16000], target[:16000], enroll[:16000]
    else:
        # Fallback synthetic
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        mix = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
        target = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        enroll = target.copy()
        return mix, target, enroll


def test_track2_pipeline_init(real_checkpoint_available):
    """Check pipeline initialization with default checkpoint."""
    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available, device="cpu")
    assert pipeline.checkpoint_path == real_checkpoint_available
    assert pipeline.device == "cpu"
    assert pipeline.extractor is None  # Lazy loading verified


def test_track2_missing_checkpoint_raises():
    """Check that invalid checkpoint path raises Track2PipelineError on extraction."""
    pipeline = Track2Pipeline(checkpoint_path=r"C:\non_existent_checkpoint_xyz.pt")
    dummy_mix = np.zeros(16000, dtype=np.float32)
    dummy_enroll = np.zeros(16000, dtype=np.float32)

    with pytest.raises(Track2PipelineError, match="checkpoint file not found"):
        pipeline.extract(mixture=dummy_mix, enrollment_audio=dummy_enroll)


def test_track2_missing_inputs_raises(real_checkpoint_available):
    """Check that calling extract without enrollment raises Track2PipelineError."""
    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available)
    dummy_mix = np.zeros(16000, dtype=np.float32)

    with pytest.raises(Track2PipelineError, match="Either enrollment_audio OR"):
        pipeline.extract(mixture=dummy_mix)


def test_track2_extraction_with_real_checkpoint(real_checkpoint_available, test_audio_data):
    """Verify PS4 extraction produces valid 16 kHz mono output with matching length."""
    mix, target, enroll = test_audio_data
    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available, device="cpu")

    result = pipeline.extract(
        mixture=mix,
        enrollment_audio=enroll,
        selected_speaker="Speaker A",
    )

    assert isinstance(result, Track2Result)
    assert result.sample_rate == 16000
    assert result.extracted_audio.ndim == 1
    assert len(result.extracted_audio) == len(mix)
    assert np.all(np.isfinite(result.extracted_audio))
    assert result.duration == 1.0
    assert result.execution_time > 0.0
    # No clean target provided, so metrics should be None
    assert result.metrics is None


def test_track2_evaluation_metrics_when_target_provided(real_checkpoint_available, test_audio_data):
    """Check SI-SDR metric computation when clean target is supplied."""
    mix, target, enroll = test_audio_data
    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available, device="cpu")

    result = pipeline.extract(
        mixture=mix,
        enrollment_audio=enroll,
        clean_target=target,
    )

    assert result.metrics is not None
    assert "mixture_si_sdr" in result.metrics
    assert "extracted_si_sdr" in result.metrics
    assert "si_sdr_improvement" in result.metrics
    assert isinstance(result.metrics["si_sdr_improvement"], float)


def test_track2_saves_audio_to_disk(tmp_path, real_checkpoint_available, test_audio_data):
    """Check saving extracted speech to disk and verifying audio format."""
    mix, _, enroll = test_audio_data
    out_file = tmp_path / "extracted_target.wav"
    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available, device="cpu")

    result = pipeline.extract(
        mixture=mix,
        enrollment_audio=enroll,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert result.output_path == str(out_file.resolve())

    loaded, loaded_sr = sf.read(str(out_file))
    assert loaded_sr == 16000
    assert loaded.ndim == 1
    assert len(loaded) == len(mix)


def test_track2_extraction_with_diarization_and_selection(real_checkpoint_available, test_audio_data):
    """Check full workflow: passing mixture, diarization result, and speaker selection."""
    mix, _, _ = test_audio_data
    # Mock diarization result with 1-second segment for Speaker B
    segments = [
        SpeakerSegment(speaker="Speaker B", start_time=0.0, end_time=1.0, duration=1.0, confidence=0.9),
    ]
    diar_result = DiarizationResult(
        segments=segments,
        speakers=["Speaker B"],
        speaker_stats={},
        total_audio_duration=1.0,
        total_speech_duration=1.0,
        sample_rate=16000,
        num_speakers=1,
    )

    pipeline = Track2Pipeline(checkpoint_path=real_checkpoint_available, device="cpu")
    result = pipeline.extract(
        mixture=mix,
        diarization_result=diar_result,
        selected_speaker="Speaker B",
    )

    assert result.selected_speaker == "Speaker B"
    assert len(result.extracted_audio) == len(mix)
    assert np.all(np.isfinite(result.extracted_audio))
