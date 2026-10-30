from pathlib import Path
import sys

# Find project root
ROOT = Path(__file__).resolve().parents[1]

# Allow Python to import from src
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.audio_loader import (
    load_audio,
    get_audio_info
)

# Path to sample audio
audio_path = ROOT / "data" / "raw" / "sample.wav"

# Load audio
audio, sample_rate = load_audio(audio_path)

# Get audio information
info = get_audio_info(audio, sample_rate)

print("\n========== ECHOLENS AUDIO INFO ==========")

for key, value in info.items():
    print(f"{key}: {value}")

print("========================================")
print("\nAudio loading successful!")