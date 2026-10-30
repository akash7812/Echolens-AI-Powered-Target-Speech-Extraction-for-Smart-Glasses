from pathlib import Path
import sys

# Find project root
ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.audio_loader import load_audio
from echolens.data.visualization import plot_spectrogram


# Load sample audio
audio_path = ROOT / "data" / "raw" / "sample.wav"

audio, sample_rate = load_audio(audio_path)

print("Displaying audio spectrograms...")

# Display spectrogram
plot_spectrogram(audio, sample_rate)