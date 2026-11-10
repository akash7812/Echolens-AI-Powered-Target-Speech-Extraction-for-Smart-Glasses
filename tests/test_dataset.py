from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.dataset import AudioDataset


def test_dataset_length():
    manifest_path = ROOT / "data" / "manifests" / "sample_manifest.csv"

    dataset = AudioDataset(
        manifest_path=manifest_path,
        project_root=ROOT
    )

    assert len(dataset) == 1


def test_dataset_audio_loading():
    manifest_path = ROOT / "data" / "manifests" / "sample_manifest.csv"

    dataset = AudioDataset(
        manifest_path=manifest_path,
        project_root=ROOT
    )

    sample = dataset[0]

    assert sample["audio"].shape == (2, 80000)
    assert sample["sample_rate"] == 16000
    assert sample["info"]["duration"] == 5.0


def test_missing_manifest_columns(tmp_path):
    manifest_path = tmp_path / "invalid_manifest.csv"

    manifest_path.write_text(
        "audio_path,sample_rate\n"
        "data/raw/sample.wav,16000\n"
    )

    with pytest.raises(ValueError, match="Missing manifest columns"):
        AudioDataset(
            manifest_path=manifest_path,
            project_root=ROOT
        )