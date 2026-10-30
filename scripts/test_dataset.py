from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echolens.data.dataset import AudioDataset


manifest_path = ROOT / "data" / "manifests" / "sample_manifest.csv"

dataset = AudioDataset(
    manifest_path=manifest_path,
    project_root=ROOT
)

print("Dataset loaded successfully!")
print("Total recordings:", len(dataset))

sample = dataset[0]

print("Audio shape:", sample["audio"].shape)
print("Sample rate:", sample["sample_rate"])
print("Audio path:", sample["audio_path"])
print("Audio information:", sample["info"])