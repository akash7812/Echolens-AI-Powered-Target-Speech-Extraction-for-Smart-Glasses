from pathlib import Path
import pandas as pd

from echolens.data.audio_loader import load_audio, get_audio_info


class AudioDataset:
    def __init__(self, manifest_path, project_root):
        self.project_root = Path(project_root)
        self.manifest = pd.read_csv(manifest_path)

        required_columns = {
            "audio_path",
            "sample_rate",
            "channels",
            "duration",
        }

        missing = required_columns - set(self.manifest.columns)

        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        audio_path = self.project_root / row["audio_path"]

        audio, sample_rate = load_audio(audio_path)

        info = get_audio_info(audio, sample_rate)

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "audio_path": str(audio_path),
            "info": info,
        }