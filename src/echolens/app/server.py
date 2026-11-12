"""
Echolens — Smart-Glasses AI Acoustic Hub & Web Application
==========================================================
Clean, Elegant, Simple & Lightweight Interface for Smart-Glasses Target Speech Extraction:
  - Track 2: Interlocutor Target Speaker Extraction (PS4 Deep BSRNN)
  - Track 1: Wearer Own-Speech Enhancement (Adaptive Harmonic Wiener Filtering)
  - Robust Speaker Diarization with Reliable Previews & Target Selection
  - Dynamic Model Switching: Fine-Tuned (Adapted) vs Original Baseline (Epoch 37)
  - In-Browser 16 kHz Mono PCM WAV Encoder for Live Microphone Input
"""

import http.server
import io
import json
from pathlib import Path
import tempfile
from typing import Optional, Dict, Any
import urllib.parse
import uuid

import numpy as np
import soundfile as sf
import torch

# Optimize PyTorch CPU execution for cloud containers (prevent thread contention)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from echolens.audio.ingestion import ingest_audio, IngestionResult
from echolens.diarization import (
    SpectralAHCDiarizer,
    SpeakerPreviewGenerator,
    SpeakerSelectionManager,
)
from echolens.pipeline.track1 import Track1Pipeline
from echolens.pipeline.track2 import Track2Pipeline


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Echolens — Smart-Glasses AI Acoustic Hub</title>
<style>
  :root {
    color-scheme: dark;
    --bg-base: #0b0f19;
    --bg-surface: #111827;
    --bg-card: #162032;
    --bg-card-hover: #1e293b;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-active: #6366f1;
    
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --emerald: #10b981;
    --emerald-dark: #059669;
    --amber: #f59e0b;
    --rose: #f43f5e;
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    background-color: var(--bg-base);
    color: var(--text-main);
    min-height: 100vh;
    padding: 32px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .app-shell {
    max-width: 860px;
    width: 100%;
  }

  /* --- HEADER --- */
  .app-header {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }
  .brand-group {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-logo {
    font-size: 30px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
  }
  .brand-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.3px;
  }
  .brand-desc {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    font-size: 11.5px;
    font-weight: 600;
    color: var(--emerald);
    font-family: ui-monospace, monospace;
  }
  .dot-pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--emerald);
  }

  /* Model Switcher */
  .model-toggle-wrap {
    display: inline-flex;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 3px;
    gap: 3px;
  }
  .model-opt-btn {
    padding: 5px 12px;
    font-size: 11.5px;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    background: transparent;
    color: var(--text-dim);
    transition: all 0.2s;
  }
  .model-opt-btn.active {
    background: var(--primary);
    color: #ffffff;
  }

  /* --- TABS --- */
  .nav-tabs {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
  }
  .nav-tab-btn {
    flex: 1;
    padding: 14px 18px;
    font-size: 14.5px;
    font-weight: 600;
    border-radius: 12px;
    cursor: pointer;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    transition: all 0.2s;
    text-align: center;
  }
  .nav-tab-btn:hover {
    color: var(--text-main);
    border-color: rgba(255, 255, 255, 0.2);
  }
  .nav-tab-btn.active {
    background: rgba(99, 102, 241, 0.12);
    border-color: var(--primary);
    color: #a5b4fc;
    font-weight: 700;
  }

  .tab-panel {
    display: none;
  }
  .tab-panel.active {
    display: block;
  }

  /* --- SECTION CARDS --- */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: border-color 0.2s;
  }
  .card:hover {
    border-color: rgba(255, 255, 255, 0.14);
  }
  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 10px;
  }
  .card-title {
    font-size: 16px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .step-badge {
    width: 24px; height: 24px;
    border-radius: 6px;
    background: var(--primary);
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }
  .card-desc {
    font-size: 13.5px;
    color: var(--text-muted);
    line-height: 1.5;
    margin-bottom: 16px;
  }

  /* --- BUTTONS & CONTROLS --- */
  .btn-group {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 9px 18px;
    font-size: 13.5px;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
    background: var(--primary);
    color: #ffffff;
  }
  .btn:hover {
    background: var(--primary-hover);
    transform: translateY(-1px);
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none;
  }
  .btn-secondary {
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-main);
    border: 1px solid var(--border-color);
  }
  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.14);
  }
  .btn-action {
    background: var(--emerald-dark);
    color: #ffffff;
  }
  .btn-action:hover {
    background: var(--emerald);
  }
  .btn-record {
    background: var(--rose);
    color: #ffffff;
  }
  .btn-record.recording {
    background: #be123c;
    animation: pulseRec 1.2s infinite;
  }
  @keyframes pulseRec {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .file-label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 9px 16px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px dashed var(--border-color);
    font-size: 13.5px;
    font-weight: 500;
    color: var(--text-main);
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .file-label:hover {
    border-color: var(--primary);
    background: rgba(99, 102, 241, 0.08);
  }
  input[type="file"] { display: none; }

  audio {
    width: 100%;
    height: 38px;
    border-radius: 8px;
    margin-top: 12px;
    outline: none;
  }

  /* --- SPEAKER CARDS GRID --- */
  .speaker-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 14px;
    margin-top: 14px;
  }
  .speaker-card {
    background: var(--bg-surface);
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .speaker-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: var(--bg-card-hover);
  }
  .speaker-card.selected {
    border-color: var(--emerald);
    background: rgba(16, 185, 129, 0.08);
  }
  .speaker-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .speaker-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-main);
  }
  .speaker-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-muted);
  }
  .speaker-card.selected .speaker-badge {
    background: var(--emerald);
    color: #064e3b;
  }
  .speaker-info {
    font-size: 12.5px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .btn-select-spk {
    width: 100%;
    padding: 8px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 600;
    border: none;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-main);
    margin-top: 10px;
    transition: all 0.15s;
  }
  .speaker-card.selected .btn-select-spk {
    background: var(--emerald);
    color: #064e3b;
    font-weight: 700;
  }

  /* --- OUTPUT COMPARISON DECK --- */
  .output-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 16px;
  }
  @media (max-width: 640px) {
    .output-grid { grid-template-columns: 1fr; }
  }
  .output-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
  }
  .output-box.success {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.05);
  }
  .output-header {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .metrics-strip {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
  }
  .metric-tag {
    padding: 6px 12px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    font-size: 12px;
    font-weight: 600;
    font-family: ui-monospace, monospace;
  }

  .status-text {
    font-size: 12px;
    color: var(--text-dim);
    font-family: ui-monospace, monospace;
  }

  footer {
    text-align: center;
    margin-top: 32px;
    font-size: 12.5px;
    color: var(--text-dim);
  }
</style>
</head>
<body>

<div class="app-shell">

  <!-- ==================== HEADER ==================== -->
  <header class="app-header">
    <div class="brand-group">
      <div class="brand-logo">👓</div>
      <div>
        <div class="brand-title">Echolens</div>
        <div class="brand-desc">AI-Powered Target Speech Extraction for Smart Glasses</div>
      </div>
    </div>

    <div class="header-badges">
      <div class="status-pill"><span class="dot-pulse"></span>ONLINE</div>
      <div class="model-toggle-wrap">
        <button id="btn-model-ft" class="model-opt-btn active" onclick="switchModel('finetuned')">⭐ Fine-Tuned</button>
        <button id="btn-model-base" class="model-opt-btn" onclick="switchModel('baseline')">Baseline (Epoch 37)</button>
      </div>
    </div>
  </header>

  <!-- ==================== NAVIGATION TABS ==================== -->
  <nav class="nav-tabs">
    <button class="nav-tab-btn active" onclick="switchTab('track2')">
      <span>🎯 Track 2: Extract Selected Speaker</span>
    </button>
    <button class="nav-tab-btn" onclick="switchTab('track1')">
      <span>✨ Track 1: Enhance My Voice</span>
    </button>
  </nav>

  <!-- ==================== TRACK 2: INTERLOCUTOR EXTRACTION ==================== -->
  <main id="panel-track2" class="tab-panel active">
    <!-- Step 1: Input -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">1</span>
          <span>Conversation Mixture Input</span>
        </div>
        <span id="t2-audio-status" class="status-text">No audio loaded</span>
      </div>
      <p class="card-desc">
        Upload a multi-talker audio mixture or record live from your smart-glasses microphone.
      </p>
      <div class="btn-group">
        <label class="file-label">
          📁 Upload Audio (WAV/MP3)
          <input type="file" id="t2-file-input" accept="audio/*" onchange="uploadMixtureTrack2(this.files[0])">
        </label>
        <button id="t2-record-btn" class="btn btn-record" onclick="toggleRecordTrack2()">🎙️ Record Microphone</button>
        <button class="btn btn-secondary" onclick="loadSampleTrack2()">⚡ Load Sample Mixture</button>
      </div>
      <audio id="t2-mixture-player" controls style="display:none;"></audio>
    </div>

    <!-- Step 2: Diarization & Speaker Selection -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">2</span>
          <span>Speaker Diarization & Target Selection</span>
        </div>
        <button id="t2-diarize-btn" class="btn" onclick="runDiarizationTrack2()" disabled>🔍 Diarize Conversation</button>
      </div>
      <p class="card-desc">
        Separates the conversation into individual speakers. Listen to preview clips and select the interlocutor to isolate.
      </p>
      <div id="t2-speakers-container" class="speaker-grid"></div>
      <p id="t2-diarize-status" style="font-size:13px; color:var(--text-muted); margin-top:8px; display:none;"></p>
    </div>

    <!-- Step 3: Extraction Output -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">3</span>
          <span>Target Extraction (PS4 Deep Neural Separator)</span>
        </div>
        <button id="t2-extract-btn" class="btn btn-action" onclick="runExtractionTrack2()" disabled>🎯 Extract Target Speaker</button>
      </div>
      <p id="t2-confirm-text" class="card-desc">
        Please select a speaker above to enable target extraction.
      </p>

      <div id="t2-output-section" style="display:none;">
        <div class="output-grid">
          <div class="output-box">
            <div class="output-header">
              <span>Original Mixture</span>
              <span style="color:var(--amber); font-size:11px;">INPUT</span>
            </div>
            <audio id="t2-orig-player" controls></audio>
          </div>

          <div class="output-box success">
            <div class="output-header">
              <span style="color:var(--emerald);">Isolated Target Voice</span>
              <span style="color:var(--emerald); font-size:11px;">EXTRACTED</span>
            </div>
            <audio id="t2-extracted-player" controls></audio>
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-top:16px; gap:10px;">
          <div id="t2-metrics-container" class="metrics-strip"></div>
          <a id="t2-download-link" class="btn" download="extracted_target_speaker.wav">⬇️ Download Clean WAV</a>
        </div>
      </div>
    </div>
  </main>

  <!-- ==================== TRACK 1: WEARER ENHANCEMENT ==================== -->
  <main id="panel-track1" class="tab-panel">
    <!-- Step 1: Wearer Profile -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">1</span>
          <span>Wearer Voice Profile (Smart-Glasses Owner)</span>
        </div>
        <span id="t1-ref-status" class="status-text">No profile registered</span>
      </div>
      <p class="card-desc">
        Enroll the glasses wearer's voice so the algorithm can isolate it from background chatter and ambient noise.
      </p>
      <div class="btn-group">
        <label class="file-label">
          📁 Upload Voice Profile
          <input type="file" id="t1-ref-file" accept="audio/*" onchange="uploadWearerRef(this.files[0])">
        </label>
        <button id="t1-ref-rec-btn" class="btn btn-record" onclick="toggleRecordWearerRef()">🎙️ Record Voice Profile</button>
        <button class="btn btn-secondary" onclick="loadSampleWearerRef()">⚡ Load Sample Profile</button>
      </div>
      <audio id="t1-ref-player" controls style="display:none;"></audio>
    </div>

    <!-- Step 2: Glasses Audio -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">2</span>
          <span>Smart-Glasses Microphone Recording</span>
        </div>
        <span id="t1-mix-status" class="status-text">No audio loaded</span>
      </div>
      <p class="card-desc">
        Audio recorded from the smart-glasses microphone containing the wearer's speech mixed with noise.
      </p>
      <div class="btn-group">
        <label class="file-label">
          📁 Upload Noisy Recording
          <input type="file" id="t1-mix-file" accept="audio/*" onchange="uploadGlassesAudio(this.files[0])">
        </label>
        <button id="t1-mix-rec-btn" class="btn btn-record" onclick="toggleRecordGlasses()">🎙️ Record Microphone</button>
        <button class="btn btn-secondary" onclick="loadSampleGlassesMix()">⚡ Load Sample Mixture</button>
      </div>
      <audio id="t1-mix-player" controls style="display:none;"></audio>
    </div>

    <!-- Step 3: Enhancement Output -->
    <div class="card">
      <div class="card-top">
        <div class="card-title">
          <span class="step-badge">3</span>
          <span>Conditioned Own-Speech Enhancement</span>
        </div>
        <button id="t1-enhance-btn" class="btn btn-action" onclick="runEnhancementTrack1()" disabled>✨ Enhance My Voice</button>
      </div>
      <p class="card-desc">
        Suppresses room reverberation, noise floor, and non-wearer speech using adaptive harmonic Wiener filtering.
      </p>

      <div id="t1-output-section" style="display:none;">
        <div class="output-grid">
          <div class="output-box">
            <div class="output-header">
              <span>Original Noisy Audio</span>
              <span style="color:var(--amber); font-size:11px;">INPUT</span>
            </div>
            <audio id="t1-orig-player" controls></audio>
          </div>

          <div class="output-box success">
            <div class="output-header">
              <span style="color:var(--emerald);">Enhanced Wearer Voice</span>
              <span style="color:var(--emerald); font-size:11px;">ENHANCED</span>
            </div>
            <audio id="t1-enh-player" controls></audio>
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-top:16px; gap:10px;">
          <div id="t1-metrics-container" class="metrics-strip"></div>
          <a id="t1-download-link" class="btn" download="enhanced_own_speech.wav">⬇️ Download Enhanced WAV</a>
        </div>
      </div>
    </div>
  </main>

  <footer>
    Echolens — Smart-Glasses Target Speech Extraction (CHiME-10 Task 3 Aligned)
  </footer>
</div>

<script>
let currentTab = 'track2';
let currentModelType = 'finetuned';

let t2MixturePath = null;
let t2Diarization = null;
let t2SelectedSpeaker = null;

let t1RefPath = null;
let t1MixPath = null;

let mediaRecorder = null;
let recordedChunks = [];

// =============================================================================
// NAVIGATION & MODEL CONTROLS
// =============================================================================
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`button[onclick="switchTab('${tab}')"]`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');
}

async function safeFetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errMsg = `Server returned status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.error) errMsg = errJson.error;
    } catch(e) {
      try {
        const text = await res.text();
        if (text) errMsg = text.slice(0, 150);
      } catch(e2) {}
    }
    throw new Error(errMsg);
  }
  return await res.json();
}

async function switchModel(modelType) {
  currentModelType = modelType;
  document.getElementById('btn-model-ft').classList.toggle('active', modelType === 'finetuned');
  document.getElementById('btn-model-base').classList.toggle('active', modelType === 'baseline');

  try {
    await fetch('/api/model/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: modelType })
    });
  } catch(e) {}
}

// =============================================================================
// IN-BROWSER 16 KHZ MONO PCM WAV ENCODER
// =============================================================================
async function convertBlobTo16kWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtxClass) throw new Error('Web Audio API not supported');
  const actx = new AudioCtxClass();
  let decoded;
  try {
    decoded = await actx.decodeAudioData(arrayBuffer.slice(0));
  } catch (decErr) {
    throw new Error('Could not decode audio data: ' + decErr.message);
  } finally {
    try { await actx.close(); } catch(e) {}
  }

  const targetSampleRate = 16000;
  const numSamples = Math.max(1, Math.round(decoded.duration * targetSampleRate));
  const offlineCtx = new OfflineAudioContext(1, numSamples, targetSampleRate);

  const bufferSource = offlineCtx.createBufferSource();
  bufferSource.buffer = decoded;
  bufferSource.connect(offlineCtx.destination);
  bufferSource.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  const channelData = renderedBuffer.getChannelData(0);

  const wavBuffer = new ArrayBuffer(44 + channelData.length * 2);
  const view = new DataView(wavBuffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + channelData.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, channelData.length * 2, true);

  let offset = 44;
  for (let i = 0; i < channelData.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}

// =============================================================================
// TRACK 2: TARGET SPEAKER EXTRACTION
// =============================================================================
async function uploadMixtureTrack2(file) {
  if (!file) return;
  let fileToUpload = file;
  if (!file.name.toLowerCase().endsWith('.wav')) {
    try {
      const wavBlob = await convertBlobTo16kWav(file);
      fileToUpload = new File([wavBlob], file.name.replace(/\.[^/.]+$/, "") + ".wav", { type: 'audio/wav' });
    } catch(e) {}
  }
  const formData = new FormData();
  formData.append('audio', fileToUpload);
  try {
    const data = await safeFetchJson('/api/ingest', { method: 'POST', body: formData });
    if (data.success) {
      t2MixturePath = data.saved_path;
      document.getElementById('t2-audio-status').innerText = `${data.duration.toFixed(1)}s (16 kHz Mono)`;
      const p = document.getElementById('t2-mixture-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      document.getElementById('t2-diarize-btn').disabled = false;
    } else {
      alert(data.error);
    }
  } catch(err) {
    alert('Upload error: ' + err.message);
  }
}

async function loadSampleTrack2() {
  try {
    const data = await safeFetchJson('/api/track2/sample', { method: 'POST' });
    if (data.success) {
      t2MixturePath = data.saved_path;
      document.getElementById('t2-audio-status').innerText = `${data.duration.toFixed(1)}s (Sample Mixture 01)`;
      const p = document.getElementById('t2-mixture-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      document.getElementById('t2-diarize-btn').disabled = false;
    } else {
      alert(data.error || 'Failed to load sample');
    }
  } catch(err) {
    alert('Sample loading error: ' + err.message);
  }
}

async function runDiarizationTrack2() {
  const btn = document.getElementById('t2-diarize-btn');
  const statusP = document.getElementById('t2-diarize-status');
  btn.disabled = true;
  btn.innerText = 'Analyzing speakers...';
  statusP.style.display = 'block';
  statusP.innerText = 'Analyzing audio with spectral AHC diarization...';

  try {
    const data = await safeFetchJson('/api/track2/diarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mixture_path: t2MixturePath })
    });
    btn.disabled = false;
    btn.innerText = '🔍 Re-diarize';

    if (data.success && data.previews) {
      t2Diarization = data.diarization;
      renderSpeakerCardsTrack2(data.previews);
      statusP.innerText = `Detected ${Object.keys(data.previews).length} speakers. Click a card to select your target.`;
    } else {
      statusP.innerText = 'Diarization failed: ' + (data.error || 'Unknown error');
      alert(data.error || 'Diarization failed');
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerText = '🔍 Diarize Conversation';
    statusP.innerText = 'Error: ' + err.message;
    alert('Error during diarization: ' + err.message);
  }
}

function renderSpeakerCardsTrack2(previews) {
  const container = document.getElementById('t2-speakers-container');
  container.innerHTML = '';
  t2SelectedSpeaker = null;
  document.getElementById('t2-extract-btn').disabled = true;
  document.getElementById('t2-confirm-text').innerText = 'Please select a speaker above to enable target extraction.';

  const avatarIcons = ['👤', '🗣️', '🧑', '👩', '🎙️'];
  let idx = 0;

  for (const [speaker, p] of Object.entries(previews)) {
    const card = document.createElement('div');
    card.className = 'speaker-card';
    card.id = `card-${speaker.replace(/\s+/g, '_')}`;
    card.onclick = () => selectSpeakerTrack2(speaker, p);

    const safeDuration = Number(p.total_detected_duration || p.duration || 0).toFixed(1);
    const turns = p.segment_count || 1;
    const icon = avatarIcons[idx % avatarIcons.length];
    idx++;

    card.innerHTML = `
      <div class="speaker-header">
        <span class="speaker-title">${icon} ${speaker}</span>
        <span class="speaker-badge" id="badge-${speaker.replace(/\s+/g, '_')}">SELECT</span>
      </div>
      <div class="speaker-info">
        Speech: <b>${safeDuration}s</b> · ${turns} turn${turns > 1 ? 's' : ''}
      </div>
      <audio controls src="/audio/${encodeURIComponent(p.filename || '')}" onclick="event.stopPropagation()"></audio>
      <button class="btn-select-spk" id="btn-spk-${speaker.replace(/\s+/g, '_')}">Select as Target</button>
    `;
    container.appendChild(card);
  }
}

function selectSpeakerTrack2(speaker, previewData) {
  t2SelectedSpeaker = speaker;
  document.querySelectorAll('.speaker-card').forEach(c => {
    c.classList.remove('selected');
    const badge = c.querySelector('.speaker-badge');
    const btn = c.querySelector('.btn-select-spk');
    if (badge) badge.innerText = 'SELECT';
    if (btn) btn.innerText = 'Select as Target';
  });

  const card = document.getElementById(`card-${speaker.replace(/\s+/g, '_')}`);
  if (card) {
    card.classList.add('selected');
    const badge = card.querySelector('.speaker-badge');
    const btn = card.querySelector('.btn-select-spk');
    if (badge) badge.innerText = 'TARGET LOCKED';
    if (btn) btn.innerText = '✓ Target Selected';
  }

  document.getElementById('t2-confirm-text').innerText = `Target Selected: ${speaker}. Click 'Extract Target Speaker' to isolate.`;
  document.getElementById('t2-extract-btn').disabled = false;
}

async function runExtractionTrack2() {
  const btn = document.getElementById('t2-extract-btn');
  btn.disabled = true;
  btn.innerText = 'Extracting via PS4 (BSRNN)...';

  try {
    const data = await safeFetchJson('/api/track2/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mixture_path: t2MixturePath,
        selected_speaker: t2SelectedSpeaker,
        model_type: currentModelType
      })
    });
    btn.disabled = false;
    btn.innerText = '🎯 Extract Target Speaker';

    if (data.success) {
      document.getElementById('t2-output-section').style.display = 'block';

      const origP = document.getElementById('t2-orig-player');
      const extP = document.getElementById('t2-extracted-player');
      origP.src = `/audio/${encodeURIComponent(data.mixture_filename)}`;
      extP.src = `/audio/${encodeURIComponent(data.extracted_filename)}`;

      document.getElementById('t2-download-link').href = `/audio/${encodeURIComponent(data.extracted_filename)}`;

      const metricsBox = document.getElementById('t2-metrics-container');
      metricsBox.innerHTML = '';
      if (data.metrics && data.metrics.si_sdr_improvement !== undefined) {
        metricsBox.innerHTML += `<div class="metric-tag">SI-SDR Improvement: +${data.metrics.si_sdr_improvement.toFixed(2)} dB</div>`;
      }
      metricsBox.innerHTML += `<div class="metric-tag">Model: ${currentModelType === 'finetuned' ? 'PS4 Fine-Tuned (Adapted)' : 'PS4 Baseline Epoch 37'}</div>`;
      metricsBox.innerHTML += `<div class="metric-tag">Inference Latency: ${data.execution_time.toFixed(2)}s (CPU)</div>`;
    } else {
      alert(data.error || 'Extraction failed');
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerText = '🎯 Extract Target Speaker';
    alert('Extraction notice: ' + err.message);
  }
}

// =============================================================================
// TRACK 1: WEARER OWN-SPEECH ENHANCEMENT
// =============================================================================
async function uploadWearerRef(file) {
  if (!file) return;
  let fileToUpload = file;
  if (!file.name.toLowerCase().endsWith('.wav')) {
    try {
      const wavBlob = await convertBlobTo16kWav(file);
      fileToUpload = new File([wavBlob], file.name.replace(/\.[^/.]+$/, "") + ".wav", { type: 'audio/wav' });
    } catch(e) {}
  }
  const formData = new FormData();
  formData.append('audio', fileToUpload);
  try {
    const data = await safeFetchJson('/api/ingest', { method: 'POST', body: formData });
    if (data.success) {
      t1RefPath = data.saved_path;
      document.getElementById('t1-ref-status').innerText = `${data.duration.toFixed(1)}s (Registered Profile)`;
      const p = document.getElementById('t1-ref-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      checkTrack1Ready();
    } else {
      alert(data.error);
    }
  } catch(err) {
    alert('Upload error: ' + err.message);
  }
}

async function loadSampleWearerRef() {
  try {
    const data = await safeFetchJson('/api/track1/sample_ref', { method: 'POST' });
    if (data.success) {
      t1RefPath = data.saved_path;
      document.getElementById('t1-ref-status').innerText = `${data.duration.toFixed(1)}s (Speaker A Profile)`;
      const p = document.getElementById('t1-ref-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      checkTrack1Ready();
    }
  } catch(err) {
    alert('Sample reference error: ' + err.message);
  }
}

async function uploadGlassesAudio(file) {
  if (!file) return;
  let fileToUpload = file;
  if (!file.name.toLowerCase().endsWith('.wav')) {
    try {
      const wavBlob = await convertBlobTo16kWav(file);
      fileToUpload = new File([wavBlob], file.name.replace(/\.[^/.]+$/, "") + ".wav", { type: 'audio/wav' });
    } catch(e) {}
  }
  const formData = new FormData();
  formData.append('audio', fileToUpload);
  try {
    const data = await safeFetchJson('/api/ingest', { method: 'POST', body: formData });
    if (data.success) {
      t1MixPath = data.saved_path;
      document.getElementById('t1-mix-status').innerText = `${data.duration.toFixed(1)}s (Glasses Recording)`;
      const p = document.getElementById('t1-mix-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      checkTrack1Ready();
    } else {
      alert(data.error);
    }
  } catch(err) {
    alert('Upload error: ' + err.message);
  }
}

async function loadSampleGlassesMix() {
  try {
    const data = await safeFetchJson('/api/track1/sample_mix', { method: 'POST' });
    if (data.success) {
      t1MixPath = data.saved_path;
      document.getElementById('t1-mix-status').innerText = `${data.duration.toFixed(1)}s (Sample Mixture 01)`;
      const p = document.getElementById('t1-mix-player');
      p.src = `/audio/${encodeURIComponent(data.filename)}`;
      p.style.display = 'block';
      checkTrack1Ready();
    }
  } catch(err) {
    alert('Sample mixture error: ' + err.message);
  }
}

function checkTrack1Ready() {
  if (t1RefPath && t1MixPath) {
    document.getElementById('t1-enhance-btn').disabled = false;
  }
}

async function runEnhancementTrack1() {
  const btn = document.getElementById('t1-enhance-btn');
  btn.disabled = true;
  btn.innerText = 'Enhancing Wearer Speech...';

  try {
    const data = await safeFetchJson('/api/track1/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mixture_path: t1MixPath,
        reference_path: t1RefPath,
      })
    });
    btn.disabled = false;
    btn.innerText = '✨ Enhance My Voice';

    if (data.success) {
      document.getElementById('t1-output-section').style.display = 'block';
      const origP = document.getElementById('t1-orig-player');
      const enhP = document.getElementById('t1-enh-player');
      origP.src = `/audio/${encodeURIComponent(data.mixture_filename)}`;
      enhP.src = `/audio/${encodeURIComponent(data.enhanced_filename)}`;

      document.getElementById('t1-download-link').href = `/audio/${encodeURIComponent(data.enhanced_filename)}`;

      const metricsBox = document.getElementById('t1-metrics-container');
      metricsBox.innerHTML = '';
      metricsBox.innerHTML += `<div class="metric-tag">Noise Suppression: +${data.noise_reduction_db.toFixed(1)} dB</div>`;
      if (data.metrics && data.metrics.si_sdr_improvement) {
        metricsBox.innerHTML += `<div class="metric-tag">SI-SDR Improvement: +${data.metrics.si_sdr_improvement.toFixed(2)} dB</div>`;
      }
      metricsBox.innerHTML += `<div class="metric-tag">Inference Latency: ${data.execution_time.toFixed(2)}s</div>`;
    } else {
      alert(data.error || 'Enhancement failed');
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerText = '✨ Enhance My Voice';
    alert('Enhancement error: ' + err.message);
  }
}

      const metricsBox = document.getElementById('t1-metrics-container');
      metricsBox.innerHTML = '';
      metricsBox.innerHTML += `<div class="metric-tag">Noise Suppression: +${data.noise_reduction_db.toFixed(1)} dB</div>`;
      if (data.metrics && data.metrics.si_sdr_improvement) {
        metricsBox.innerHTML += `<div class="metric-tag">SI-SDR Improvement: +${data.metrics.si_sdr_improvement.toFixed(2)} dB</div>`;
      }
      metricsBox.innerHTML += `<div class="metric-tag">Inference Latency: ${data.execution_time.toFixed(2)}s</div>`;
    } else {
      alert(data.error || 'Enhancement failed');
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerText = '✨ Enhance My Voice';
    alert('Enhancement error: ' + err.message);
  }
}

// =============================================================================
// RECORDING WITH NATIVE PCM CONVERSION
// =============================================================================
async function toggleRecordTrack2() {
  const btn = document.getElementById('t2-record-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    btn.classList.remove('recording');
    btn.innerText = 'Processing Audio...';
    mediaRecorder.stop();
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          if (rawBlob.size < 100) {
            throw new Error('Recording was too short or empty. Please speak into your microphone.');
          }
          const wavBlob = await convertBlobTo16kWav(rawBlob);
          await uploadMixtureTrack2(new File([wavBlob], 'glasses_mic_recording.wav', { type: 'audio/wav' }));
        } catch (convErr) {
          alert('Microphone recording error: ' + convErr.message);
        } finally {
          btn.classList.remove('recording');
          btn.innerText = '🎙️ Record Microphone';
          stream.getTracks().forEach(t => t.stop());
        }
      };
      mediaRecorder.start(250);
      btn.classList.add('recording');
      btn.innerText = '⏹️ Stop Recording';
    } catch(err) {
      alert('Microphone access denied: ' + err.message);
    }
  }
}

async function toggleRecordWearerRef() {
  const btn = document.getElementById('t1-ref-rec-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    btn.classList.remove('recording');
    btn.innerText = 'Processing Audio...';
    mediaRecorder.stop();
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          if (rawBlob.size < 100) {
            throw new Error('Recording was too short or empty. Please speak into your microphone.');
          }
          const wavBlob = await convertBlobTo16kWav(rawBlob);
          await uploadWearerRef(new File([wavBlob], 'wearer_ref.wav', { type: 'audio/wav' }));
        } catch (convErr) {
          alert('Microphone recording error: ' + convErr.message);
        } finally {
          btn.classList.remove('recording');
          btn.innerText = '🎙️ Record Voice Profile';
          stream.getTracks().forEach(t => t.stop());
        }
      };
      mediaRecorder.start(250);
      btn.classList.add('recording');
      btn.innerText = '⏹️ Stop Recording';
    } catch(err) {
      alert('Microphone access denied: ' + err.message);
    }
  }
}

async function toggleRecordGlasses() {
  const btn = document.getElementById('t1-mix-rec-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    btn.classList.remove('recording');
    btn.innerText = 'Processing Audio...';
    mediaRecorder.stop();
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          if (rawBlob.size < 100) {
            throw new Error('Recording was too short or empty. Please speak into your microphone.');
          }
          const wavBlob = await convertBlobTo16kWav(rawBlob);
          await uploadGlassesAudio(new File([wavBlob], 'glasses_mic.wav', { type: 'audio/wav' }));
        } catch (convErr) {
          alert('Microphone recording error: ' + convErr.message);
        } finally {
          btn.classList.remove('recording');
          btn.innerText = '🎙️ Record Microphone';
          stream.getTracks().forEach(t => t.stop());
        }
      };
      mediaRecorder.start(250);
      btn.classList.add('recording');
      btn.innerText = '⏹️ Stop Recording';
    } catch(err) {
      alert('Microphone access denied: ' + err.message);
    }
  }
}
</script>
</body>
</html>
"""


class EcholensAppServer:
    """Manages the backend session state and audio storage for the web application."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        checkpoint_path: Optional[str] = None,
    ):
        if storage_dir is None:
            self.storage_dir = Path(tempfile.mkdtemp(prefix="echolens_app_"))
        else:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.audio_cache_dir = self.storage_dir / "audio_cache"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

        # Multi-platform data and model file discovery (Windows, Linux, Docker, Render)
        def find_ckpt(name: str, win_fallback: str) -> Path:
            import os
            env_path = os.environ.get("ECHOLENS_CHECKPOINT_PATH")
            candidates = []
            if env_path and name in os.path.basename(env_path):
                candidates.append(Path(env_path))
            candidates.extend([
                Path("models") / name,
                Path("/app/models") / name,
                Path(__file__).resolve().parents[3] / "models" / name,
                Path(win_fallback),
            ])
            for p in candidates:
                if p and p.is_file():
                    return p
            return Path(win_fallback)

        self.finetuned_path = find_ckpt("checkpoint_finetuned.pt", r"C:\Echolens_Models\checkpoint_finetuned.pt")
        self.baseline_path = find_ckpt("checkpoint_epoch037.pt", r"C:\Echolens_Models\checkpoint_epoch037.pt")

        if checkpoint_path:
            chosen_ckpt = checkpoint_path
        elif self.finetuned_path.is_file():
            chosen_ckpt = str(self.finetuned_path)
        else:
            chosen_ckpt = str(self.baseline_path)

        # Initialize Pipelines
        self.track1_pipeline = Track1Pipeline()
        self.track2_pipeline = Track2Pipeline(checkpoint_path=chosen_ckpt, device="cpu")
        self.diarizer = SpectralAHCDiarizer()
        self.preview_generator = SpeakerPreviewGenerator()
        self.selection_manager = SpeakerSelectionManager()

        # Pre-load PS4 model into memory to eliminate first-request latency / timeouts
        try:
            self.track2_pipeline._ensure_model_loaded()
            print(f"[Echolens] PS4 model pre-loaded into memory from {chosen_ckpt}")
        except Exception as e:
            print(f"[Echolens] Model pre-load note: {e}")

        self.current_diarization = None
        self.current_mixture_audio = None
        self.current_model_type = "finetuned" if "finetuned" in str(chosen_ckpt) else "baseline"

    def set_active_model(self, model_type: str):
        """Switch between fine-tuned and baseline checkpoints."""
        if model_type == "baseline" and self.baseline_path.is_file():
            self.track2_pipeline = Track2Pipeline(checkpoint_path=str(self.baseline_path), device="cpu")
            self.current_model_type = "baseline"
        elif self.finetuned_path.is_file():
            self.track2_pipeline = Track2Pipeline(checkpoint_path=str(self.finetuned_path), device="cpu")
            self.current_model_type = "finetuned"


def find_data_file(rel_path: str) -> Optional[Path]:
    """Resolve sample and reference data files across local Windows, repo root, and Docker /app paths."""
    clean_rel = rel_path.replace("\\", "/")
    candidates = [
        Path(clean_rel),
        Path(__file__).resolve().parents[3] / clean_rel,
        Path("/app") / clean_rel,
        Path(r"C:\Echolens") / clean_rel,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def make_request_handler(app_state: EcholensAppServer):
    """Factory creating the HTTP request handler bound to the server instance."""

    class EcholensHTTPHandler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, data: Dict[str, Any]):
            resp = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)

        def do_HEAD(self):
            parsed = urllib.parse.urlparse(self.path)
            self.send_response(200)
            if parsed.path in ["/", "/index.html"]:
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/health":
                self.send_header("Content-Type", "application/json")
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/health":
                self._send_json(200, {"status": "ok", "app": "echolens"})
                return

            if parsed.path == "/" or parsed.path == "/index.html":
                page = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return

            if parsed.path.startswith("/audio/"):
                filename = urllib.parse.unquote(parsed.path[7:])
                file_path = app_state.audio_cache_dir / filename
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                else:
                    self.send_error(404, "Audio file not found")
                    return

            self.send_error(404, "Endpoint not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)

            # Model switch
            if parsed.path == "/api/model/select":
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                m_type = payload.get("model_type", "finetuned")
                app_state.set_active_model(m_type)
                self._send_json(200, {"success": True, "active_model": app_state.current_model_type})
                return

            # 1. Ingest Audio Endpoint
            if parsed.path == "/api/ingest":
                content_type = self.headers.get("Content-Type", "")
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                # Multipart parser for raw file uploads
                if "multipart/form-data" in content_type:
                    boundary = content_type.split("boundary=")[-1].strip().encode()
                    parts = body.split(b"--" + boundary)
                    audio_bytes = None
                    for part in parts:
                        if b"filename=" in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                raw_content = part[header_end + 4:]
                                if raw_content.endswith(b"\r\n"):
                                    audio_bytes = raw_content[:-2]
                                else:
                                    audio_bytes = raw_content
                                break
                else:
                    audio_bytes = body

                if not audio_bytes:
                    self._send_json(400, {"success": False, "error": "No audio data received."})
                    return

                try:
                    filename = f"ingest_{uuid.uuid4().hex[:8]}.wav"
                    save_path = app_state.audio_cache_dir / filename
                    ingest_res = ingest_audio(audio_bytes, save_path=save_path)
                    app_state.current_mixture_audio = ingest_res.audio
                    self._send_json(200, {
                        "success": True,
                        "filename": filename,
                        "saved_path": str(save_path),
                        "duration": ingest_res.duration,
                        "sample_rate": ingest_res.sample_rate,
                    })
                except Exception as e:
                    self._send_json(500, {"success": False, "error": str(e)})
                return

            # 2. Track 2 Sample Mixture
            if parsed.path == "/api/track2/sample":
                sample_p = find_data_file("data/processed/training_samples/mixture_01.wav")
                if sample_p and sample_p.is_file():
                    audio, sr = sf.read(str(sample_p), dtype="float32")
                    filename = f"sample_t2_{uuid.uuid4().hex[:8]}.wav"
                    dest = app_state.audio_cache_dir / filename
                    sf.write(str(dest), audio, sr)
                    app_state.current_mixture_audio = audio
                    self._send_json(200, {
                        "success": True,
                        "filename": filename,
                        "saved_path": str(dest),
                        "duration": round(len(audio) / sr, 2),
                    })
                else:
                    self._send_json(404, {"success": False, "error": "Sample mixture file not found on server."})
                return

            # 3. Track 2 Diarization & Previews
            if parsed.path == "/api/track2/diarize":
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                mix_path = payload.get("mixture_path")

                try:
                    audio, sr = sf.read(mix_path, dtype="float32")
                    app_state.current_mixture_audio = audio
                    diar_res = app_state.diarizer.diarize(audio, sample_rate=sr)
                    app_state.current_diarization = diar_res

                    # Generate previews
                    previews = app_state.preview_generator.generate_previews(
                        audio=audio,
                        diarization_result=diar_res,
                        output_dir=app_state.audio_cache_dir,
                    )

                    preview_dict = {}
                    for spk, p in previews.items():
                        fname = Path(p.saved_path).name if p.saved_path else ""
                        p_dict = p.to_dict()
                        p_dict["filename"] = fname
                        preview_dict[spk] = p_dict

                    self._send_json(200, {
                        "success": True,
                        "diarization": diar_res.to_dict(),
                        "previews": preview_dict,
                    })
                except Exception as e:
                    self._send_json(500, {"success": False, "error": str(e)})
                return

            # 4. Track 2 Extraction (PS4)
            if parsed.path == "/api/track2/extract":
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                mix_path = payload.get("mixture_path")
                selected_speaker = payload.get("selected_speaker")

                try:
                    audio, sr = sf.read(mix_path, dtype="float32")
                    out_fname = f"extracted_ps4_{uuid.uuid4().hex[:8]}.wav"
                    out_path = app_state.audio_cache_dir / out_fname

                    target_p = find_data_file("data/processed/training_samples/target_01.wav")
                    clean_target = None
                    if target_p and target_p.is_file() and selected_speaker == "Speaker A":
                        t_audio, _ = sf.read(str(target_p), dtype="float32")
                        if len(t_audio) == len(audio):
                            clean_target = t_audio

                    diar_res = app_state.current_diarization
                    if diar_res is None:
                        diar_res = app_state.diarizer.diarize(audio, sample_rate=sr)
                        app_state.current_diarization = diar_res

                    res = app_state.track2_pipeline.extract(
                        mixture=audio,
                        diarization_result=diar_res,
                        selected_speaker=selected_speaker,
                        output_path=out_path,
                        clean_target=clean_target,
                    )

                    self._send_json(200, {
                        "success": True,
                        "extracted_filename": out_fname,
                        "mixture_filename": Path(mix_path).name,
                        "metrics": res.metrics,
                        "execution_time": res.execution_time,
                    })
                except Exception as e:
                    self._send_json(500, {"success": False, "error": str(e)})
                return

            # 5. Track 1 Sample Endpoints
            if parsed.path == "/api/track1/sample_ref":
                ref_p = find_data_file("data/raw/speaker_a_new_enrollment.wav")
                if ref_p and ref_p.is_file():
                    audio, sr = sf.read(str(ref_p), dtype="float32")
                    fname = f"sample_ref_{uuid.uuid4().hex[:8]}.wav"
                    dest = app_state.audio_cache_dir / fname
                    sf.write(str(dest), audio, sr)
                    self._send_json(200, {
                        "success": True,
                        "filename": fname,
                        "saved_path": str(dest),
                        "duration": round(len(audio) / sr, 2),
                    })
                else:
                    self._send_json(404, {"success": False, "error": "Sample reference file not found on server."})
                return

            if parsed.path == "/api/track1/sample_mix":
                mix_p = find_data_file("data/processed/training_samples/mixture_01.wav")
                if mix_p and mix_p.is_file():
                    audio, sr = sf.read(str(mix_p), dtype="float32")
                    fname = f"sample_mix_{uuid.uuid4().hex[:8]}.wav"
                    dest = app_state.audio_cache_dir / fname
                    sf.write(str(dest), audio, sr)
                    self._send_json(200, {
                        "success": True,
                        "filename": fname,
                        "saved_path": str(dest),
                        "duration": round(len(audio) / sr, 2),
                    })
                else:
                    self._send_json(404, {"success": False, "error": "Sample mixture file not found on server."})
                return

            # 6. Track 1 Enhancement
            if parsed.path == "/api/track1/enhance":
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                mix_path = payload.get("mixture_path")
                ref_path = payload.get("reference_path")

                try:
                    mix_audio, sr = sf.read(mix_path, dtype="float32")
                    ref_audio, _ = sf.read(ref_path, dtype="float32")

                    target_p = find_data_file("data/processed/training_samples/target_01.wav")
                    clean_target = None
                    if target_p and target_p.is_file():
                        t_audio, _ = sf.read(str(target_p), dtype="float32")
                        if len(t_audio) == len(mix_audio):
                            clean_target = t_audio

                    out_fname = f"enhanced_t1_{uuid.uuid4().hex[:8]}.wav"
                    out_path = app_state.audio_cache_dir / out_fname

                    res = app_state.track1_pipeline.enhance(
                        mixture=mix_audio,
                        wearer_reference=ref_audio,
                        clean_target=clean_target,
                        output_path=out_path,
                    )

                    self._send_json(200, {
                        "success": True,
                        "enhanced_filename": out_fname,
                        "mixture_filename": Path(mix_path).name,
                        "noise_reduction_db": res.noise_reduction_db,
                        "metrics": res.metrics,
                        "execution_time": res.execution_time,
                    })
                except Exception as e:
                    self._send_json(500, {"success": False, "error": str(e)})
                return

            self.send_error(404, "API route not found")

    return EcholensHTTPHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    storage_dir: Optional[Path] = None,
    checkpoint_path: Optional[str] = None,
) -> http.server.HTTPServer:
    """Create and return an instantiated ThreadingHTTPServer bound to the requested host and port."""
    app_state = EcholensAppServer(storage_dir=storage_dir, checkpoint_path=checkpoint_path)
    handler = make_request_handler(app_state)
    return http.server.ThreadingHTTPServer((host, port), handler)
