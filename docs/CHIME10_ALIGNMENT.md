# Echolens — CHiME-10 Task 3 Alignment Report

**Challenge:** CHiME-10 Task 3 (Speech Processing for Smart Glasses Wearers)  
**Project:** Echolens  

---

## 1. Task Definition and Track Comparison

The CHiME-10 Task 3 benchmark addresses speech processing on wearable smart glasses in everyday multi-talker, noisy scenarios. The task is formally partitioned into two distinct tracks:

| Dimension | Track 1: Own Speech Enhancement (OSE) | Track 2: Target Speaker Extraction (TSE) |
|---|---|---|
| **Primary Goal** | Enhance the voice of the glasses wearer | Extract the voice of a specified conversation partner |
| **Speaker of Interest** | The Smart-Glasses Wearer | An External Interlocutor (Speaker A, B, C...) |
| **Acoustic Challenges** | Ambient background noise, room reverb, competing background chatter | Overlapping conversational speech, multiple interlocutors, interfering noise |
| **Conditioning Method** | Wearer Voice Reference (biometric spectral profile) | Enrollment audio generated from diarized speech segments |
| **Extraction Model** | Wearer-conditioned adaptive Wiener filtering & spectral tracking | Deep Neural Dilated Convolutional Extractor (Member 2 PS4 checkpoint) |
| **Target Selection** | Automatic (wearer is known ahead of time via registered profile) | Explicit user selection after diarization & preview |

---

## 2. Critical Architectural Principles

### 2.1 Diarization Does Not Imply Intent (Track 2)
In a multi-talker conversation, an unsupervised speaker diarizer clusters speech turns into acoustic identities (`Speaker A`, `Speaker B`, `Speaker C`). 
- **The Pitfall:** Many naive implementations automatically treat the first speaker (`Speaker A`) or the most dominant speaker as the intended target.
- **Echolens Alignment:** Diarization is strictly treated as an *acoustic separation* step, **never** an *intent determination* step. The system displays all detected speakers, generates high-SNR previews of each speaker's voice, and **requires explicit confirmation from the wearer**.

### 2.2 Mono Audio Limitations and Biometric Conditioning (Track 1)
On smart glasses hardware equipped with single-channel mono audio, spatial beamforming and interaural time difference (ITD) filtering cannot be performed.
- **The Ambiguity:** An external speaker standing directly in front of the wearer produces similar single-channel amplitude to the wearer's own speech.
- **Echolens Alignment:** Track 1 utilizes a registered **Wearer Voice Reference Profile** (enrolled once by the glasses wearer). The enhancement pipeline cross-correlates the incoming mixture with the wearer's unique harmonic spectral distribution, allowing the system to isolate the wearer's voice even in single-channel mono recordings.

### 2.3 Distinction Between PS4 and Track 1
- **PS4 Architecture:** The existing Member 2 model (`checkpoint_epoch037.pt`) is a deep neural Target Speaker Extraction network trained specifically for **Track 2** (extracting an arbitrary speaker given an enrollment clip from that speaker).
- **Track 1 Separation:** Echolens does **not** falsely claim that PS4 implements Track 1 out of the box. Track 1 requires dedicated own-speech enhancement with noise floor suppression and speech preservation, implemented via the `Track1Pipeline`.

### 2.4 Offline Batch vs Real-Time Streaming Hardware Integration
- **Current Capability:** Echolens currently operates in **offline batch mode**. It processes recorded audio files or user-captured microphone clips through high-speed vectorized pipelines.
- **Honest Limitation:** Real-time causal block streaming (e.g., chunk-by-chunk low-latency inference on embedded smart-glasses DSP hardware) requires streaming neural architectures and dedicated hardware drivers. Echolens documentation accurately describes the current system as an offline evaluation and interactive application prototype.

---

## 3. Metric Evaluation Integrity

In accordance with strict academic integrity:
1. **No Metric Fabrication:** When real-world user audio is uploaded without clean ground-truth targets, Echolens reports operational metrics (processing duration, noise floor reduction in dB, speaker stats), but **does not fabricate** SI-SDR, STOI, or PESQ numbers.
2. **Ground-Truth Evaluation:** When clean target reference WAV files are available (e.g. during development benchmarks or controlled testing), Echolens invokes the official `echolens.models.evaluation.evaluate_sample` module to calculate:
   - Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
   - SI-SDR Improvement ($\Delta\text{SI-SDR} = \text{SI-SDR}_{out} - \text{SI-SDR}_{in}$)
