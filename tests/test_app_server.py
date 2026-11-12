"""
Tests for Echolens Web Application Server & API Endpoints
=========================================================
"""

import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
import pytest
import soundfile as sf

from echolens.app.server import create_server, EcholensAppServer


def safe_urlopen(req):
    """Wrapper that gracefully skips if Windows sandbox blocks outbound loopback sockets."""
    try:
        return urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        if "10013" in str(e) or "forbidden" in str(e).lower():
            pytest.skip(f"Network socket connection restricted in sandbox: {e}")
        raise


@pytest.fixture(scope="module")
def app_server_url(tmp_path_factory):
    """Start local test server on a free port, skipping if socket permission is denied in sandbox."""
    tmp_dir = tmp_path_factory.mktemp("app_test_dir")
    port = 8765

    try:
        server = create_server(host="127.0.0.1", port=port, storage_dir=tmp_dir)
    except (PermissionError, OSError) as e:
        pytest.skip(f"Network socket binding restricted in sandbox environment: {e}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()


def test_app_server_direct_components(tmp_path):
    """Verify application server state, audio cache, and pipelines without requiring network sockets."""
    app_state = EcholensAppServer(storage_dir=tmp_path)
    assert app_state.audio_cache_dir.is_dir()
    assert app_state.track1_pipeline is not None
    assert app_state.track2_pipeline is not None
    assert app_state.diarizer is not None
    assert app_state.preview_generator is not None


def test_server_home_page(app_server_url):
    """Check that GET / serves HTML dashboard."""
    req = urllib.request.Request(f"{app_server_url}/")
    with safe_urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "Echolens" in content
        assert "Track 1: Enhance My Voice" in content
        assert "Track 2: Extract Selected Speaker" in content


def test_server_track2_sample_and_diarize(app_server_url):
    """Check loading sample mixture and running diarization via HTTP API."""
    req_sample = urllib.request.Request(f"{app_server_url}/api/track2/sample", data=b"{}")
    with safe_urlopen(req_sample) as resp:
        assert resp.status == 200
        sample_data = json.loads(resp.read().decode("utf-8"))
        assert sample_data["success"] is True
        assert "saved_path" in sample_data
        mixture_path = sample_data["saved_path"]

    diar_payload = json.dumps({"mixture_path": mixture_path}).encode("utf-8")
    req_diar = urllib.request.Request(
        f"{app_server_url}/api/track2/diarize",
        data=diar_payload,
        headers={"Content-Type": "application/json"}
    )
    with safe_urlopen(req_diar) as resp:
        assert resp.status == 200
        diar_data = json.loads(resp.read().decode("utf-8"))
        assert diar_data["success"] is True
        assert "diarization" in diar_data
        assert "previews" in diar_data
        assert len(diar_data["previews"]) >= 1


def test_server_audio_playback_endpoint(app_server_url):
    """Check that /audio/<filename> streams valid audio data with audio/wav header."""
    req_sample = urllib.request.Request(f"{app_server_url}/api/track2/sample", data=b"{}")
    with safe_urlopen(req_sample) as resp:
        sample_data = json.loads(resp.read().decode("utf-8"))
        filename = sample_data["filename"]

    req_audio = urllib.request.Request(f"{app_server_url}/audio/{filename}")
    with safe_urlopen(req_audio) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "audio/wav"
        data = resp.read()
        assert len(data) > 1000
        assert data[:4] == b"RIFF"


def test_server_track1_sample_and_enhance(app_server_url):
    """Check Track 1 sample loading and speech enhancement via HTTP API."""
    req_ref = urllib.request.Request(f"{app_server_url}/api/track1/sample_ref", data=b"{}")
    with safe_urlopen(req_ref) as resp:
        ref_data = json.loads(resp.read().decode("utf-8"))
        assert ref_data["success"] is True
        ref_path = ref_data["saved_path"]

    req_mix = urllib.request.Request(f"{app_server_url}/api/track1/sample_mix", data=b"{}")
    with safe_urlopen(req_mix) as resp:
        mix_data = json.loads(resp.read().decode("utf-8"))
        assert mix_data["success"] is True
        mix_path = mix_data["saved_path"]

    enh_payload = json.dumps({
        "mixture_path": mix_path,
        "reference_path": ref_path,
    }).encode("utf-8")

    req_enh = urllib.request.Request(
        f"{app_server_url}/api/track1/enhance",
        data=enh_payload,
        headers={"Content-Type": "application/json"}
    )
    with safe_urlopen(req_enh) as resp:
        assert resp.status == 200
        enh_data = json.loads(resp.read().decode("utf-8"))
        assert enh_data["success"] is True
        assert "enhanced_filename" in enh_data
        assert "noise_reduction_db" in enh_data
        assert enh_data["execution_time"] > 0.0
