"""
Tests for Track 1 Wearer Own-Speech Enhancement Pipeline
========================================================
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from echolens.pipeline.track1 import (
    Track1Pipeline,
    Track1Result,
    Track1PipelineError,
)


@pytest.fixture
def synthetic_track1_data():
    """
    Generate synthetic smart-glasses audio:
    - Bursty wearer speech: 400 Hz harmonic series with speech envelope
    - Ambient noise: Gaussian noise
    - Mixture: Wearer speech + Ambient noise
    - Wearer reference: Separate utterance of 400 Hz harmonic series
    """
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Bursty wearer speech with natural pauses
    envelope = np.clip(np.sin(2 * np.pi * 2 * t), 0, 1)
    wearer_clean = (envelope * (0.5 * np.sin(2 * np.pi * 400 * t) + 0.25 * np.sin(2 * np.pi * 800 * t))).astype(np.float32)

    # Ambient noise
    noise = (0.05 * np.random.randn(len(t))).astype(np.float32)

    # Mixture
    mixture = (wearer_clean + noise).astype(np.float32)

    # Reference recording (1.5s)
    t_ref = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    wearer_ref = (0.5 * np.sin(2 * np.pi * 400 * t_ref) + 0.25 * np.sin(2 * np.pi * 800 * t_ref)).astype(np.float32)

    return mixture, wearer_clean, wearer_ref, sr


def test_track1_enhancement_success(synthetic_track1_data):
    """Check running Track 1 enhancement produces valid 16 kHz mono audio."""
    mix, clean, ref, sr = synthetic_track1_data
    pipeline = Track1Pipeline(sample_rate=sr)

    result = pipeline.enhance(mixture=mix, wearer_reference=ref)

    assert isinstance(result, Track1Result)
    assert result.sample_rate == 16000
    assert result.enhanced_audio.ndim == 1
    assert len(result.enhanced_audio) == len(mix)
    assert np.all(np.isfinite(result.enhanced_audio))
    assert result.duration == 2.0
    assert result.execution_time > 0.0
    # No clean target provided, metrics should be None
    assert result.metrics is None


def test_track1_noise_reduction(synthetic_track1_data):
    """Verify that background noise floor is measurably attenuated."""
    mix, clean, ref, sr = synthetic_track1_data
    pipeline = Track1Pipeline(sample_rate=sr)

    result = pipeline.enhance(mixture=mix, wearer_reference=ref)

    assert result.noise_reduction_db > 0.0


def test_track1_evaluation_when_target_provided(synthetic_track1_data):
    """Check that SI-SDR metrics are computed when ground-truth clean target is supplied."""
    mix, clean, ref, sr = synthetic_track1_data
    pipeline = Track1Pipeline(sample_rate=sr)

    result = pipeline.enhance(
        mixture=mix,
        wearer_reference=ref,
        clean_target=clean,
    )

    assert result.metrics is not None
    assert "mixture_si_sdr" in result.metrics
    assert "extracted_si_sdr" in result.metrics
    assert "si_sdr_improvement" in result.metrics
    assert "noise_reduction_db" in result.metrics
    assert result.metrics["si_sdr_improvement"] > 0.0


def test_track1_saves_audio_to_disk(tmp_path, synthetic_track1_data):
    """Check saving enhanced audio to disk as 16-bit PCM WAV."""
    mix, _, ref, sr = synthetic_track1_data
    out_file = tmp_path / "enhanced_own_speech.wav"
    pipeline = Track1Pipeline(sample_rate=sr)

    result = pipeline.enhance(
        mixture=mix,
        wearer_reference=ref,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert result.output_path == str(out_file.resolve())

    loaded, loaded_sr = sf.read(str(out_file))
    assert loaded_sr == 16000
    assert loaded.ndim == 1
    assert len(loaded) == len(mix)


def test_track1_empty_mixture_raises():
    """Check that empty mixture raises Track1PipelineError."""
    pipeline = Track1Pipeline()
    empty = np.array([], dtype=np.float32)
    ref = np.ones(16000, dtype=np.float32)

    with pytest.raises(Track1PipelineError, match="non-empty 1D mono"):
        pipeline.enhance(mixture=empty, wearer_reference=ref)


def test_track1_empty_reference_raises():
    """Check that empty wearer reference raises Track1PipelineError."""
    pipeline = Track1Pipeline()
    mix = np.ones(16000, dtype=np.float32)
    empty = np.array([], dtype=np.float32)

    with pytest.raises(Track1PipelineError, match="non-empty 1D mono"):
        pipeline.enhance(mixture=mix, wearer_reference=empty)


def test_track1_real_dataset_sample():
    """Check Track 1 enhancement using real project dataset files."""
    mix_p = Path(r"C:\Echolens\data\processed\training_samples\mixture_01.wav")
    ref_p = Path(r"C:\Echolens\data\raw\speaker_a_new_enrollment.wav")
    target_p = Path(r"C:\Echolens\data\processed\training_samples\target_01.wav")

    if mix_p.is_file() and ref_p.is_file() and target_p.is_file():
        mix, sr = sf.read(str(mix_p), dtype="float32")
        ref, _ = sf.read(str(ref_p), dtype="float32")
        target, _ = sf.read(str(target_p), dtype="float32")

        pipeline = Track1Pipeline(sample_rate=sr)
        result = pipeline.enhance(
            mixture=mix,
            wearer_reference=ref,
            clean_target=target,
        )

        assert result.sample_rate == 16000
        assert len(result.enhanced_audio) == len(mix)
        assert np.all(np.isfinite(result.enhanced_audio))
        assert result.metrics is not None
