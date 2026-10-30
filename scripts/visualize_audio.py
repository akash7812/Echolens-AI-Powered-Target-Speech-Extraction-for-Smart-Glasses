from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.audio_loader import load_audio
from echolens.data.visualization import plot_waveform


audio_path = ROOT / "data" / "raw" / "sample.wav"

audio, sample_rate = load_audio(audio_path)

print("Displaying audio waveforms...")

plot_waveform(audio, sample_rate)