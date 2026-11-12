#!/usr/bin/env python3
"""
Echolens — PS4 Model Domain Adaptation & Fine-Tuning Pipeline
=============================================================
Fine-tunes the PS4 Target Speaker Extraction checkpoint (BSRNN)
to adapt from clean anechoic datasets to real-world smart-glasses /
microphone acoustics (reverberation, laptop/fan noise, and gain variation).

Features:
  - Dynamic on-the-fly mixture generation with randomized SNRs (-5 dB to +8 dB).
  - Acoustic simulation: Synthetic Room Impulse Response (RIR) reverberation.
  - Realistic ambient noise injection (fan, pink, and diffuse noise).
  - Random gain scaling (0.2x to 1.0x) matching real microphone dynamics.
  - Preserves original checkpoint; saves to C:\\Echolens_Models\\checkpoint_finetuned.pt.
  - Efficient CPU execution via frozen speaker encoder and mask-adapter fine-tuning.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import fftconvolve

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from echolens.models.inference import build_model, load_checkpoint
from echolens.models.evaluation import si_sdr


DEFAULT_CHECKPOINT_IN = r"C:\Echolens_Models\checkpoint_epoch037.pt"
DEFAULT_CHECKPOINT_OUT = r"C:\Echolens_Models\checkpoint_finetuned.pt"
SAMPLE_RATE = 16000


def generate_synthetic_rir(duration_sec: float = 0.25, rt60: float = 0.20, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate synthetic room impulse response (RIR) modeling acoustic reverberation.
    Decay rate tau is chosen such that energy drops by 60 dB at rt60.
    """
    num_samples = int(duration_sec * sr)
    t = np.arange(num_samples) / sr
    decay_rate = 3.0 * np.log(10.0) / max(0.05, rt60)
    envelope = np.exp(-decay_rate * t)
    noise = np.random.randn(num_samples).astype(np.float32)
    rir = envelope * noise
    rir[0] = 1.0  # Direct acoustic path
    return (rir / (np.linalg.norm(rir) + 1e-8)).astype(np.float32)


def generate_realistic_noise(num_samples: int) -> np.ndarray:
    """Generate ambient room noise (pink + low-frequency rumble like laptop fans)."""
    white = np.random.randn(num_samples).astype(np.float32)
    # Low-pass filter to simulate fan/ambient rumble
    b = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], dtype=np.float32)
    pink = np.convolve(white, b, mode="same")
    norm = np.max(np.abs(pink)) + 1e-8
    return (pink / norm).astype(np.float32)


class DynamicMixtureDataset:
    """
    Generates dynamic augmented training samples on-the-fly from available clean speech.
    """

    def __init__(self, raw_dir: Path, segment_seconds: float = 2.0, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.seg_len = int(segment_seconds * sr)

        target_p = raw_dir / "speaker_a_new_target.wav"
        interferer_p = raw_dir / "speaker_b.wav"
        enroll_p = raw_dir / "speaker_a_new_enrollment.wav"

        self.target_audio, _ = sf.read(str(target_p), dtype="float32")
        self.interferer_audio, _ = sf.read(str(interferer_p), dtype="float32")
        self.enroll_audio, _ = sf.read(str(enroll_p), dtype="float32")

    def sample_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce a randomized training batch:
          mixture: (B, T)
          enrollment: (B, T_enroll)
          clean_target: (B, T)
        """
        mixtures, enrollments, targets = [], [], []

        for _ in range(batch_size):
            # 1. Random segment from target speech
            max_t_idx = max(0, len(self.target_audio) - self.seg_len)
            t_idx = np.random.randint(0, max_t_idx + 1)
            clean_tgt = self.target_audio[t_idx : t_idx + self.seg_len].copy()

            # 2. Random segment from interferer speech
            max_i_idx = max(0, len(self.interferer_audio) - self.seg_len)
            i_idx = np.random.randint(0, max_i_idx + 1)
            interferer = self.interferer_audio[i_idx : i_idx + self.seg_len].copy()

            # 3. Random segment for enrollment (2.0s - 3.0s)
            enroll_len = min(len(self.enroll_audio), int(np.random.uniform(2.0, 3.0) * self.sr))
            max_e_idx = max(0, len(self.enroll_audio) - enroll_len)
            e_idx = np.random.randint(0, max_e_idx + 1)
            enroll = self.enroll_audio[e_idx : e_idx + enroll_len].copy()

            # 4. Randomized target-to-interferer SNR (-4 dB to +6 dB)
            tgt_rms = np.sqrt(np.mean(clean_tgt ** 2)) + 1e-8
            int_rms = np.sqrt(np.mean(interferer ** 2)) + 1e-8
            desired_snr_db = np.random.uniform(-4.0, 6.0)
            snr_scale = (tgt_rms / int_rms) * (10.0 ** (-desired_snr_db / 20.0))
            mix = clean_tgt + snr_scale * interferer

            # 5. Acoustic Augmentation: Reverberation (50% probability)
            if np.random.rand() > 0.5:
                rt60 = float(np.random.uniform(0.12, 0.35))
                rir = generate_synthetic_rir(duration_sec=0.2, rt60=rt60, sr=self.sr)
                mix = fftconvolve(mix, rir, mode="full")[: self.seg_len].astype(np.float32)

            # 6. Acoustic Augmentation: Ambient Room / Laptop Fan Noise (SNR: 12 dB to 25 dB)
            if np.random.rand() > 0.3:
                noise = generate_realistic_noise(self.seg_len)
                noise_snr_db = np.random.uniform(12.0, 25.0)
                mix_rms = np.sqrt(np.mean(mix ** 2)) + 1e-8
                noise_scale = mix_rms * (10.0 ** (-noise_snr_db / 20.0))
                mix = mix + noise_scale * noise

            # 7. Random Gain Variation (0.2x to 1.0x)
            gain = float(np.random.uniform(0.2, 1.0))
            mix = mix * gain
            clean_tgt = clean_tgt * gain
            enroll = enroll * float(np.random.uniform(0.4, 1.0))

            # Prevent clipping
            max_val = max(1.0, float(np.max(np.abs(mix))))
            mix = mix / max_val
            clean_tgt = clean_tgt / max_val
            enroll = enroll / max(1.0, float(np.max(np.abs(enroll))))

            mixtures.append(mix)
            enrollments.append(enroll)
            targets.append(clean_tgt)

        mix_tensor = torch.from_numpy(np.stack(mixtures)).float()
        enroll_tensor = torch.from_numpy(np.stack(enrollments)).float()
        tgt_tensor = torch.from_numpy(np.stack(targets)).float()

        return mix_tensor, enroll_tensor, tgt_tensor


def sisdr_loss(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss."""
    target_z = target - torch.mean(target, dim=-1, keepdim=True)
    est_z = est - torch.mean(est, dim=-1, keepdim=True)

    dot = torch.sum(est_z * target_z, dim=-1, keepdim=True) + eps
    norm = torch.sum(target_z ** 2, dim=-1, keepdim=True) + eps
    alpha = dot / norm

    s_target = alpha * target_z
    e_noise = est_z - s_target

    snr = (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return -10.0 * torch.log10(snr).mean()


def run_finetuning(
    checkpoint_in: str = DEFAULT_CHECKPOINT_IN,
    checkpoint_out: str = DEFAULT_CHECKPOINT_OUT,
    num_steps: int = 30,
    learning_rate: float = 1e-4,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes domain-adaptation fine-tuning on the PS4 model.
    """
    device = torch.device(device_name)
    raw_dir = PROJECT_ROOT / "data/raw"

    print("=================================================================")
    print(" Echolens PS4 Model Fine-Tuning & Domain Adaptation")
    print("=================================================================")
    print(f" Source Checkpoint : {checkpoint_in}")
    print(f" Output Checkpoint : {checkpoint_out}")
    print(f" Target Steps      : {num_steps}")
    print(f" Learning Rate     : {learning_rate}")
    print(f" Device            : {device}")

    # 1. Load Model
    model = build_model(device)
    load_checkpoint(checkpoint_in, model, device)
    model.train()

    # 2. Freeze Speaker Encoder & Separator Backbone, Fine-tune Mask Adaptation Heads
    for p in model.parameters():
        p.requires_grad = False

    for p in model.mask.parameters():
        p.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable Parameters (Mask Adaptation Heads): {trainable_count:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-5,
    )

    dataset = DynamicMixtureDataset(raw_dir=raw_dir, segment_seconds=2.0, sr=SAMPLE_RATE)

    # 3. Baseline Validation Before Fine-Tuning
    model.eval()
    val_mix, val_enr, val_tgt = dataset.sample_batch(batch_size=1)
    with torch.no_grad():
        baseline_est, _ = model(val_mix.to(device), val_enr.to(device))
        baseline_sisdr = si_sdr(val_tgt.squeeze().numpy(), baseline_est.squeeze().cpu().numpy())
    print(f"\n Initial Validation SI-SDR : {baseline_sisdr:.2f} dB\n")

    # 4. Fine-Tuning Loop
    model.train()
    history = []
    t_start = time.time()

    print("--- Starting Fine-Tuning Iterations ---")
    for step in range(1, num_steps + 1):
        step_t0 = time.time()

        mix_b, enr_b, tgt_b = dataset.sample_batch(batch_size=1)
        mix_b = mix_b.to(device)
        enr_b = enr_b.to(device)
        tgt_b = tgt_b.to(device)

        optimizer.zero_grad()
        est_b, _ = model(mix_b, enr_b)
        loss = sisdr_loss(est_b, tgt_b)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mask.parameters(), max_norm=5.0)
        optimizer.step()

        step_elapsed = time.time() - step_t0
        current_sisdr = -loss.item()
        history.append(current_sisdr)

        if step % 5 == 0 or step == 1 or step == num_steps:
            print(
                f"  Step {step:02d}/{num_steps:02d} | "
                f"Loss: {loss.item():.3f} | "
                f"Batch SI-SDR: {current_sisdr:.2f} dB | "
                f"Step Time: {step_elapsed:.2f}s"
            )

    total_time = time.time() - t_start
    print(f"\n Fine-Tuning Completed in {total_time:.1f}s ({total_time / 60.0:.2f} min)")

    # 5. Final Validation
    model.eval()
    with torch.no_grad():
        final_est, _ = model(val_mix.to(device), val_enr.to(device))
        final_sisdr = si_sdr(val_tgt.squeeze().numpy(), final_est.squeeze().cpu().numpy())

    improvement = final_sisdr - baseline_sisdr
    print(f" Final Validation SI-SDR   : {final_sisdr:.2f} dB")
    print(f" Validation Improvement    : {improvement:+.2f} dB")

    # 6. Save Fine-Tuned Checkpoint
    out_path = Path(checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_save = {
        "model": model.state_dict(),
        "step": step,
        "base_epoch": 37,
        "adapted_for": "Real-room acoustics & microphone dynamics",
        "baseline_sisdr": baseline_sisdr,
        "final_sisdr": final_sisdr,
    }
    torch.save(ckpt_save, str(out_path))
    print(f"\n Checkpoint Saved Successfully to: {out_path}")
    print("=================================================================")

    return {
        "success": True,
        "saved_path": str(out_path),
        "steps": num_steps,
        "baseline_sisdr": round(float(baseline_sisdr), 2),
        "final_sisdr": round(float(final_sisdr), 2),
        "improvement_db": round(float(improvement), 2),
        "training_time_sec": round(float(total_time), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Echolens PS4 Fine-Tuning Pipeline")
    parser.add_argument("--checkpoint-in", type=str, default=DEFAULT_CHECKPOINT_IN, help="Path to initial checkpoint")
    parser.add_argument("--checkpoint-out", type=str, default=DEFAULT_CHECKPOINT_OUT, help="Path to save fine-tuned checkpoint")
    parser.add_argument("--steps", type=int, default=25, help="Number of fine-tuning optimization steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device ('cpu' or 'cuda')")

    args = parser.parse_args()
    res = run_finetuning(
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        num_steps=args.steps,
        learning_rate=args.lr,
        device_name=args.device,
    )
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
