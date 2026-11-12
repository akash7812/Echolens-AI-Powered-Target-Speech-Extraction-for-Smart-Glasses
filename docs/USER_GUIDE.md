# Echolens — User Guide & Manual

**Project:** Echolens — AI-Powered Target Speech Extraction for Smart Glasses  
**Interface:** Interactive Web Dashboard & Command-Line Interface (CLI)  

---

## 1. Quick Start

### 1.1 Environment Setup
Ensure the Python environment is activated:
```powershell
cd C:\Echolens
.\.venv\Scripts\Activate.ps1
```

Verify system diagnostics and PS4 model checkpoint status:
```bash
python -m echolens info
```

### 1.2 Starting the Web Dashboard
Launch the interactive web application server:
```bash
python -m echolens server --port 8000
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 2. Web Application Interface Guide

The web dashboard provides two dedicated operational modes via top-level navigation tabs:
- **Track 2: Extract Selected Speaker** (CHiME-10 Interlocutor Extraction)
- **Track 1: Enhance My Voice** (CHiME-10 Wearer Own-Speech Enhancement)

---

### 2.1 Track 2 Workflow: Extracting an Interlocutor

#### Step 1: Load Conversation Audio
- **Upload File**: Click `📁 Upload Mixture WAV/MP3` to load any recorded conversation.
- **Microphone**: Click `🎙️ Record Microphone` to capture live audio from your microphone or smart-glasses input.
- **Sample Audio**: Click `⚡ Load Sample Mixture` to test with preloaded multi-speaker conversation data (`mixture_01.wav`).

#### Step 2: Speaker Diarization & Audio Previews
- Click `🔍 Diarize Conversation`.
- The system segments speech turns and clusters them into distinct acoustic identities: `Speaker A`, `Speaker B`, etc.
- Each speaker is presented with a dedicated card showing:
  - Total detected speech duration.
  - Active segment count.
  - Audio player previewing the cleanest speech segments of that speaker.

#### Step 3: Explicit Interlocutor Selection
- Click on the card of the speaker you wish to isolate.
- The card highlights with a bright cyan border and displays a checkmark indicator (`SELECTED`).
- **Safety Guarantee**: The system will never assume `Speaker A` by default; target extraction remains disabled until you explicitly select a speaker.
- If a speaker has insufficient speech for reliable extraction ($< 1.5$ seconds), an advisory warning is displayed.

#### Step 4: PS4 Target Extraction
- Click `🎯 Extract Target Speaker`.
- The application automatically extracts a 1.0s–4.0s enrollment clip from the selected speaker's segments and feeds it along with the mixture into the deep neural PS4 model (`checkpoint_epoch037.pt`).
- Once processing finishes:
  - **Original Mixture Player** and **Extracted Voice Player** appear side-by-side for instantaneous comparison.
  - Runtime and SI-SDR metrics (if ground truth target exists) are displayed.
  - Click `⬇️ Download 16 kHz WAV` to export the extracted speech file.

---

### 2.2 Track 1 Workflow: Enhancing Wearer's Voice

#### Step 1: Register Wearer Voice Profile
- Single-channel mono smart-glasses audio requires a voice profile of the glasses wearer to differentiate their voice from room noise and surrounding chatter.
- Click `📁 Upload Wearer Voice Sample`, `🎙️ Record Voice Profile`, or `⚡ Load Sample Profile` (`speaker_a_new_enrollment.wav`).

#### Step 2: Load Noisy Glasses Recording
- Upload, record, or load a sample noisy conversation recording (`mixture_01.wav`).

#### Step 3: Run Enhancement
- Click `✨ Enhance My Voice`.
- The pipeline estimates the ambient noise power spectral density, tracks the wearer's harmonic profile, and applies a conditioned Wiener filter to attenuate noise floor and non-wearer speech.
- Listen to the enhanced speech side-by-side with the noisy input and download the enhanced WAV file.

---

## 3. Command-Line Interface (CLI) Guide

Echolens provides a complete command-line interface for headless execution and automated evaluation.

### 3.1 System Info & Diagnostics
```bash
python -m echolens info
```
Outputs Python version, PyTorch version, CUDA availability, and PS4 checkpoint size and location.

### 3.2 Track 2: Interlocutor Extraction via CLI
**Using an existing enrollment file:**
```bash
python -m echolens track2 \
  --mixture data/processed/training_samples/mixture_01.wav \
  --enrollment data/raw/speaker_a_new_enrollment.wav \
  --target data/processed/training_samples/target_01.wav \
  --output extracted_output.wav
```

**Using speaker diarization and automatic enrollment:**
```bash
python -m echolens track2 \
  --mixture data/processed/training_samples/mixture_01.wav \
  --speaker "Speaker A" \
  --output extracted_output.wav
```

**Output machine-readable JSON results:**
```bash
python -m echolens track2 \
  --mixture data/processed/training_samples/mixture_01.wav \
  --enrollment data/raw/speaker_a_new_enrollment.wav \
  --json
```

### 3.3 Track 1: Wearer Enhancement via CLI
```bash
python -m echolens track1 \
  --mixture data/processed/training_samples/mixture_01.wav \
  --reference data/raw/speaker_a_new_enrollment.wav \
  --target data/processed/training_samples/target_01.wav \
  --output enhanced_wearer.wav
```

---

## 4. REST API Reference

The Echolens HTTP server exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the HTML5 web dashboard. |
| `GET` | `/audio/<filename>` | Streams 16 kHz WAV audio files with `audio/wav` headers. |
| `POST` | `/api/ingest` | Ingests raw audio uploads or multipart files, returning normalized 16 kHz WAV. |
| `POST` | `/api/track2/sample` | Stages preloaded multi-speaker mixture audio. |
| `POST` | `/api/track2/diarize` | Runs speaker diarization and generates speaker preview clips. |
| `POST` | `/api/track2/extract` | Executes PS4 target speaker extraction on selected speaker. |
| `POST` | `/api/track1/sample_ref` | Stages sample wearer voice profile. |
| `POST` | `/api/track1/sample_mix` | Stages sample smart-glasses mixture. |
| `POST` | `/api/track1/enhance` | Runs wearer own-speech enhancement pipeline. |
