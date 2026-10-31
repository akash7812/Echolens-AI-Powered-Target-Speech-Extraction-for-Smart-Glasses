# Echolens — Audio Data Pipeline Documentation

## 1. Project Overview

**Project:** Echolens — AI-Powered Target Speech Extraction for Smart Glasses

**Module:** Audio Data Pipeline

**Responsible Team Member:** Member 1

The audio data pipeline prepares speech recordings for the target speaker extraction system.

The goal of the project is to isolate a selected speaker's voice from a mixture containing multiple speakers.

The audio pipeline is responsible for loading, preprocessing, mixing, organizing, and validating audio data before it is used by the machine learning model.

## 2. Audio Format

The pipeline uses the following audio specifications:

| Property | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Audio format | WAV |
| Processing data type | Float32 |
| Training sample duration | 4 seconds |
| Number of development samples | 4 |

These specifications provide a consistent format for downstream model development.

## 3. Input Audio Recordings

The pipeline currently uses three main recordings:

| Recording | Purpose |
|---|---|
| `speaker_a_new_target.wav` | Clean speech from the target speaker |
| `speaker_a_new_enrollment.wav` | Reference recording used to identify the target speaker |
| `speaker_b.wav` | Interfering speaker recording |

The recordings are stored in:

`data/raw/`

The target speaker and enrollment recordings belong to Speaker A, while Speaker B provides interfering speech.

## 4. Audio Preprocessing

The preprocessing module is located at:

`src/echolens/data/preprocessing.py`

Audio preprocessing prepares recordings for consistent processing.

The pipeline supports operations such as sample-rate conversion and mono audio preparation.

The audio loader is located at:

`src/echolens/data/audio_loader.py`

These modules provide reusable functions for loading and preparing audio recordings.

## 5. Audio Mixture Generation

The speaker mixture generation script is located at:

`scripts/create_speaker_mixture.py`

The script combines speech from Speaker A and Speaker B to create a mixed recording.

The clean target speech is also preserved for comparison with the mixture.

The resulting development recordings include:

- `data/processed/speaker_mixture_new.wav`
- `data/processed/speaker_a_target_new.wav`

These recordings can be used to test the target speaker extraction pipeline.

## 6. Training Sample Generation

The training sample generation script is located at:

`scripts/create_training_samples.py`

It divides the available recordings into four non-overlapping, four-second segments.

For each segment, the script creates:

1. A mixture containing Speaker A and Speaker B.
2. A corresponding clean target recording containing Speaker A.

Both recordings use the same scaling factor when the mixture needs normalization, preserving their relative amplitude relationship.

The generated files are stored in:

`data/processed/training_samples/`

### Generated Files

| Mixture | Target |
|---|---|
| `mixture_01.wav` | `target_01.wav` |
| `mixture_02.wav` | `target_02.wav` |
| `mixture_03.wav` | `target_03.wav` |
| `mixture_04.wav` | `target_04.wav` |

Each generated file has:

- A sample rate of 16,000 Hz.
- One audio channel.
- A duration of four seconds.

**Important limitation:** These four samples are generated from the same two speaker recordings. They are intended for development and pipeline verification, not for training a production-quality speech extraction model.

## 7. Target Speaker Dataset Loader

The dataset loader is located at:

`src/echolens/data/target_speaker_dataset.py`

The `TargetSpeakerDataset` class loads the mixture, enrollment, and clean target recordings.

It supports two modes:

**Single-sample mode**

```python
dataset = TargetSpeakerDataset(project_root)
```

This loads the original processed mixture and target pair.

**Multiple-sample mode**

```python
dataset = TargetSpeakerDataset(
    project_root,
    use_training_samples=True
)
```

This loads the generated training sample pairs.

Each dataset item returns:

```python
{
    "mixture": mixture,
    "enrollment": enrollment,
    "target": target,
    "sample_rate": 16000
}
```

The loader checks file existence, sample rate, mono channels, audio length consistency, and invalid numerical values.

## 8. Dataset CSV Manifest

The manifest generation script is located at:

`scripts/create_dataset_manifest.py`

Run it using:

```powershell
python scripts/create_dataset_manifest.py
```

The generated manifest is saved at:

`data/processed/training_manifest.csv`

The CSV contains four columns:

| Column | Description |
|---|---|
| `sample_id` | Unique identifier for the development sample |
| `mixture` | Relative path to the mixed audio |
| `enrollment` | Relative path to the target speaker reference audio |
| `target` | Relative path to the clean target audio |

The manifest contains four sample entries.

Paths are stored relative to the project root to make the dataset easier to organize and share.

## 9. Dataset Validation

The dataset has undergone the following checks:

- All four CSV sample entries were detected.
- All referenced audio files exist.
- All checked audio files use a 16,000 Hz sample rate.
- All checked audio files are mono.
- No checked audio files are empty.
- Every mixture and its clean target have matching frame counts.

### Validation Results

```text
Total samples: 4
Missing audio files: 0

Samples checked: 4
Validation errors: 0

All audio files passed format validation!
```

These checks verify file availability and audio format consistency. They do not measure perceptual speech quality or model performance.

## 10. Automated Testing

The project uses `pytest` for automated testing.

Tests are located in:

`tests/`

The target speaker dataset tests are located at:

`tests/test_target_speaker_dataset.py`

The test suite covers dataset loading, sample consistency, missing files, invalid indices, and multiple-sample loading.

Run all tests from the project root using:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Latest verified result:

```text
16 passed
```

## 11. How to Reproduce the Development Dataset

From the Echolens project root, activate the Python virtual environment and ensure the required dependencies are installed.

Make sure the three required input recordings are available under `data/raw/`.

Generate the speaker mixtures:

```powershell
python scripts/create_speaker_mixture.py
```

Generate the four development samples:

```powershell
python scripts/create_training_samples.py
```

Generate the CSV manifest:

```powershell
python scripts/create_dataset_manifest.py
```

Run automated tests:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The generated WAV files and CSV manifest should then be available under `data/processed/`.

## 12. Handover to the Model Development Team

The model development team can use the dataset loader to obtain:

- **Mixture:** Speech containing the target and interfering speakers.
- **Enrollment:** Reference speech identifying the target speaker.
- **Target:** Clean speech used as the desired extraction output.

The development dataset contains four samples, each four seconds long.

Before serious model training or performance evaluation, the team should expand the dataset with additional speakers, recordings, acoustic conditions, and suitable training, validation, and test splits.

The current dataset demonstrates that the audio preparation and loading workflow is functioning.

## 13. Module Summary

The Member 1 audio pipeline includes:

- Audio loading.
- Audio preprocessing.
- Audio visualization.
- Audio augmentation utilities.
- Speaker mixture generation.
- Development sample generation.
- Target speaker dataset loading.
- CSV manifest generation.
- Dataset format validation.
- Automated tests.

**Current status:** The development audio data pipeline is implemented and has passed the recorded validation checks.