import csv
from pathlib import Path

# Find the Echolens project folder
project_root = Path(__file__).resolve().parents[1]

# Input and output locations
samples_dir = project_root / "data/processed/training_samples"
enrollment_path = (
    project_root / "data/raw/speaker_a_new_enrollment.wav"
)

manifest_path = (
    project_root / "data/processed/training_manifest.csv"
)

# Check enrollment recording
if not enrollment_path.is_file():
    raise FileNotFoundError(
        f"Enrollment recording not found: {enrollment_path}"
    )

# Find generated mixtures
mixture_files = sorted(samples_dir.glob("mixture_*.wav"))

if not mixture_files:
    raise FileNotFoundError(
        f"No mixture recordings found in {samples_dir}"
    )

# Create one CSV row for each mixture-target pair
rows = []

for mixture_path in mixture_files:
    sample_id = mixture_path.stem.replace("mixture_", "")

    target_path = (
        samples_dir / f"target_{sample_id}.wav"
    )

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Target recording not found: {target_path}"
        )

    rows.append({
        "sample_id": sample_id,
        "mixture": mixture_path.relative_to(project_root).as_posix(),
        "enrollment": enrollment_path.relative_to(project_root).as_posix(),
        "target": target_path.relative_to(project_root).as_posix(),
    })

# Save CSV manifest
manifest_path.parent.mkdir(parents=True, exist_ok=True)

with manifest_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sample_id",
            "mixture",
            "enrollment",
            "target",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print("Dataset manifest created successfully!")
print(f"Total samples: {len(rows)}")
print(f"Manifest saved to: {manifest_path}")