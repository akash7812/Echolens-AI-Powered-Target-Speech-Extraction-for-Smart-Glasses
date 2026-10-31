from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from echolens.data.target_speaker_dataset import TargetSpeakerDataset


@pytest.fixture
def sample_audio_files(tmp_path):
    """Create temporary audio files for testing."""

    sample_rate = 16000

    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    # Create 1 second of test audio
    mixture = np.zeros(sample_rate, dtype=np.float32)
    target = np.zeros(sample_rate, dtype=np.float32)

    # Enrollment can have a different duration
    enrollment = np.zeros(sample_rate // 2, dtype=np.float32)

    sf.write(
        processed_dir / "speaker_mixture_new.wav",
        mixture,
        sample_rate,
    )

    sf.write(
        processed_dir / "speaker_a_target_new.wav",
        target,
        sample_rate,
    )

    sf.write(
        raw_dir / "speaker_a_new_enrollment.wav",
        enrollment,
        sample_rate,
    )

    return tmp_path


def test_dataset_loads_audio(sample_audio_files):
    """Check that all three recordings load correctly."""

    dataset = TargetSpeakerDataset(sample_audio_files)

    assert len(dataset) == 1

    sample = dataset[0]

    assert sample["sample_rate"] == 16000
    assert len(sample["mixture"]) == 16000
    assert len(sample["target"]) == 16000
    assert len(sample["enrollment"]) == 8000


def test_mixture_and_target_lengths_match(sample_audio_files):
    """Check that mixture and target have matching lengths."""

    dataset = TargetSpeakerDataset(sample_audio_files)

    sample = dataset[0]

    assert sample["mixture"].shape == sample["target"].shape


def test_missing_audio_file(tmp_path):
    """Check that missing recordings raise an error."""

    with pytest.raises(FileNotFoundError):
        TargetSpeakerDataset(tmp_path)


def test_invalid_dataset_index(sample_audio_files):
    """Check that invalid dataset indexes raise an error."""

    dataset = TargetSpeakerDataset(sample_audio_files)

    with pytest.raises(IndexError):
        dataset[1]