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


# --------------------------------------------------
# NEW TESTS FOR MULTIPLE TRAINING SAMPLES
# --------------------------------------------------


@pytest.fixture
def multiple_audio_samples(sample_audio_files):
    """Create four temporary mixture-target pairs."""

    samples_dir = (
        sample_audio_files
        / "data"
        / "processed"
        / "training_samples"
    )

    samples_dir.mkdir(parents=True)

    sample_rate = 16000
    duration_samples = sample_rate * 4

    for index in range(1, 5):
        mixture = np.full(
            duration_samples,
            0.1 * index,
            dtype=np.float32,
        )

        target = np.full(
            duration_samples,
            0.05 * index,
            dtype=np.float32,
        )

        sf.write(
            samples_dir / f"mixture_{index:02d}.wav",
            mixture,
            sample_rate,
        )

        sf.write(
            samples_dir / f"target_{index:02d}.wav",
            target,
            sample_rate,
        )

    return sample_audio_files


def test_multiple_samples_load(multiple_audio_samples):
    """Check that all four samples are detected."""

    dataset = TargetSpeakerDataset(
        multiple_audio_samples,
        use_training_samples=True,
    )

    assert len(dataset) == 4

    for index in range(4):
        sample = dataset[index]

        assert sample["sample_rate"] == 16000
        assert len(sample["mixture"]) == 64000
        assert len(sample["target"]) == 64000
        assert len(sample["enrollment"]) == 8000


def test_multiple_samples_are_distinct(multiple_audio_samples):
    """Check that different indexes load different recordings."""

    dataset = TargetSpeakerDataset(
        multiple_audio_samples,
        use_training_samples=True,
    )

    first_sample = dataset[0]
    second_sample = dataset[1]

    assert not np.array_equal(
        first_sample["mixture"],
        second_sample["mixture"],
    )

    assert not np.array_equal(
        first_sample["target"],
        second_sample["target"],
    )