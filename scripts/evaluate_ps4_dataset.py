
import csv
from pathlib import Path

import numpy as np

from src.echolens.data.target_speaker_dataset import TargetSpeakerDataset
from src.echolens.models.echolens_pipeline import EcholensPipeline


ROOT = Path(__file__).resolve().parents[1]

dataset = TargetSpeakerDataset(
    project_root=ROOT,
    use_training_samples=True
)

pipeline = EcholensPipeline(
    checkpoint_path=r"C:\Echolens_Models\checkpoint_epoch037.pt",
    device="cpu"
)

output_dir = ROOT / "data/processed/ps4_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

results = []

for index in range(len(dataset)):

    sample = dataset[index]

    output_path = output_dir / f"extracted_{index + 1:02d}.wav"

    result = pipeline.process_sample(
        sample,
        output_path=output_path
    )

    metrics = result["metrics"]

    row = {
        "sample": index + 1,
        "mixture_si_sdr": metrics["mixture_si_sdr"],
        "extracted_si_sdr": metrics["extracted_si_sdr"],
        "si_sdr_improvement": metrics["si_sdr_improvement"]
    }

    results.append(row)

    print(
        f"Sample {index + 1}: "
        f"Mixture={row['mixture_si_sdr']:.3f} dB | "
        f"Extracted={row['extracted_si_sdr']:.3f} dB | "
        f"Improvement={row['si_sdr_improvement']:.3f} dB"
    )

csv_path = ROOT / "results/ps4_multi_sample_evaluation.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

mean_improvement = np.mean([
    row["si_sdr_improvement"] for row in results
])

print("\n--- FINAL DEVELOPMENT SUMMARY ---")
print("Samples evaluated:", len(results))
print(f"Mean SI-SDR improvement: {mean_improvement:.3f} dB")
print("Results saved:", csv_path)
print("Audio saved:", output_dir)
