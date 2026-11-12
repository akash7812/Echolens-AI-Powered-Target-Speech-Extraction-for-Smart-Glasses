"""
Tests for Echolens CLI Interface
================================
"""

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from echolens.cli import main, create_parser


def _extract_json(text: str) -> dict:
    """Helper to extract json object from stdout even if log messages exist."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"No JSON object found in output: {text}")
    return json.loads(text[start : end + 1])


@pytest.fixture
def cli_audio_files(tmp_path):
    """Generate synthetic test WAV files for CLI testing."""
    sr = 16000
    t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False, dtype=np.float32)

    # Clean target
    clean = 0.6 * np.sin(2 * np.pi * 300 * t)
    target_path = tmp_path / "target.wav"
    sf.write(str(target_path), clean, sr)

    # Reference / Enrollment
    ref = 0.5 * np.sin(2 * np.pi * 300 * t)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), ref, sr)

    # Mixture
    interferer = 0.5 * np.sin(2 * np.pi * 700 * t)
    noise = 0.05 * np.random.randn(len(t)).astype(np.float32)
    mixture = clean + interferer + noise
    mix_path = tmp_path / "mixture.wav"
    sf.write(str(mix_path), mixture, sr)

    return {
        "target": target_path,
        "ref": ref_path,
        "mixture": mix_path,
    }


def test_cli_help(capsys):
    """Test displaying top-level CLI help message."""
    ret = main(["--help"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Echolens" in captured.out
    assert "track1" in captured.out
    assert "track2" in captured.out
    assert "server" in captured.out


def test_cli_info(capsys):
    """Test info command diagnostics."""
    ret = main(["info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Echolens System Diagnostics" in captured.out
    assert "PyTorch Version" in captured.out
    assert "PS4 Checkpoint" in captured.out


def test_cli_track1(cli_audio_files, tmp_path, capsys):
    """Test running Track 1 enhancement via CLI with JSON output."""
    out_wav = tmp_path / "t1_out.wav"
    argv = [
        "track1",
        "--mixture", str(cli_audio_files["mixture"]),
        "--reference", str(cli_audio_files["ref"]),
        "--target", str(cli_audio_files["target"]),
        "--output", str(out_wav),
        "--json",
    ]
    ret = main(argv)
    assert ret == 0
    assert out_wav.exists()

    captured = capsys.readouterr()
    res = _extract_json(captured.out)
    assert res["success"] is True
    assert "noise_reduction_db" in res
    assert "metrics" in res
    assert res["metrics"]["si_sdr_improvement"] is not None


def test_cli_track2_direct_enrollment(cli_audio_files, tmp_path, capsys):
    """Test running Track 2 extraction via CLI using direct enrollment audio."""
    out_wav = tmp_path / "t2_out.wav"
    argv = [
        "track2",
        "--mixture", str(cli_audio_files["mixture"]),
        "--enrollment", str(cli_audio_files["ref"]),
        "--target", str(cli_audio_files["target"]),
        "--output", str(out_wav),
        "--json",
    ]
    ret = main(argv)
    assert ret == 0
    assert out_wav.exists()

    captured = capsys.readouterr()
    res = _extract_json(captured.out)
    assert res["success"] is True
    assert "metrics" in res
    assert res["metrics"]["si_sdr_improvement"] is not None
