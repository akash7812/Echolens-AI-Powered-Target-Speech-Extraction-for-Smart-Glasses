"""
Echolens — Offline Speaker Diarization Module
=============================================
Performs 100% offline speaker diarization using:
  1. Fast NumPy short-time energy Voice Activity Detection (VAD).
  2. Torchaudio MFCC acoustic embedding extraction.
  3. Scikit-learn Agglomerative Hierarchical Clustering (AHC).
  4. Chronological speaker labeling: "Speaker A", "Speaker B", "Speaker C", etc.

Zero external network calls, zero Hugging Face tokens, fully compatible with Python 3.12 and NumPy 2.x.
"""

from typing import List, Dict, Any, Optional, Tuple
import string

import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .base import (
    BaseDiarizer,
    SpeakerSegment,
    SpeakerStats,
    DiarizationResult,
)


class SpectralAHCDiarizer(BaseDiarizer):
    """
    High-performance offline speaker diarizer using Torchaudio acoustic embeddings
    and Scikit-learn Agglomerative Hierarchical Clustering.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: int = 25,
        frame_hop_ms: int = 10,
        window_duration: float = 1.0,
        window_hop: float = 0.5,
        energy_threshold: float = 0.015,
        min_speech_duration: float = 0.25,
        merge_gap_threshold: float = 0.35,
        max_speakers: int = 4,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_hop = int(sample_rate * frame_hop_ms / 1000)
        self.window_duration = window_duration
        self.window_hop = window_hop
        self.window_samples = int(sample_rate * window_duration)
        self.window_hop_samples = int(sample_rate * window_hop)
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        self.merge_gap_threshold = merge_gap_threshold
        self.max_speakers = max_speakers

        # Initialize torchaudio MFCC extractor
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": self.frame_length,
                "hop_length": self.frame_hop,
                "n_mels": 26,
            },
        )

    def _voice_activity_detection(
        self,
        audio: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Detect active speech regions based on short-term energy.
        Returns a list of (start_sample, end_sample) tuples for active speech.
        """
        if len(audio) < self.frame_length:
            return []

        # Vectorized short-time RMS energy computation using NumPy sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(audio, self.frame_length)[::self.frame_hop]
        frame_energy = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=-1))

        # Dynamic thresholding based on noise floor
        noise_floor = float(np.percentile(frame_energy, 15))
        peak_energy = float(np.percentile(frame_energy, 95))

        if peak_energy < 1e-5:
            # Entire signal is near-silent
            return []

        adaptive_threshold = max(
            self.energy_threshold,
            noise_floor + 0.15 * (peak_energy - noise_floor),
        )

        speech_frames = frame_energy >= adaptive_threshold

        # Smoothing over 5 frames
        kernel_size = 5
        smoothed = np.copy(speech_frames)
        for i in range(len(smoothed)):
            start_k = max(0, i - kernel_size // 2)
            end_k = min(len(smoothed), i + kernel_size // 2 + 1)
            smoothed[i] = np.sum(speech_frames[start_k:end_k]) > (kernel_size // 2)

        # Convert frame flags to continuous sample boundaries
        active_regions: List[Tuple[int, int]] = []
        in_speech = False
        start_frame = 0

        for frame_idx, is_speech in enumerate(smoothed):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = frame_idx
            elif not is_speech and in_speech:
                in_speech = False
                start_sample = start_frame * self.frame_hop
                end_sample = min(len(audio), frame_idx * self.frame_hop + self.frame_length)
                if (end_sample - start_sample) >= int(self.min_speech_duration * self.sample_rate):
                    active_regions.append((start_sample, end_sample))

        if in_speech:
            start_sample = start_frame * self.frame_hop
            end_sample = len(audio)
            if (end_sample - start_sample) >= int(self.min_speech_duration * self.sample_rate):
                active_regions.append((start_sample, end_sample))

        # Bridge small inter-speech gaps (< 0.25s)
        merged_regions: List[Tuple[int, int]] = []
        max_gap_samples = int(0.25 * self.sample_rate)

        for region in active_regions:
            if not merged_regions:
                merged_regions.append(region)
            else:
                last_start, last_end = merged_regions[-1]
                cur_start, cur_end = region
                if cur_start - last_end <= max_gap_samples:
                    merged_regions[-1] = (last_start, cur_end)
                else:
                    merged_regions.append(region)

        return merged_regions

    def _extract_embedding(self, chunk: np.ndarray) -> np.ndarray:
        """
        Extract speaker acoustic feature embedding from an audio chunk using Torchaudio.
        Computes 13 MFCCs, mean and standard deviation pooling, and L2 normalization.
        """
        with torch.no_grad():
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            if tensor.shape[-1] < self.frame_length:
                tensor = torch.nn.functional.pad(
                    tensor, (0, self.frame_length - tensor.shape[-1])
                )
            mfcc = self.mfcc_extractor(tensor).squeeze(0).numpy()  # (13, frames)

        # Statistical pooling across time
        mean_feat = np.mean(mfcc, axis=1)
        std_feat = np.std(mfcc, axis=1)

        embedding = np.concatenate([mean_feat, std_feat])
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _cluster_windows(
        self,
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster feature embeddings using Agglomerative Hierarchical Clustering."""
        n_samples = len(embeddings)

        if n_samples == 0:
            return np.array([], dtype=int)
        if n_samples == 1:
            return np.array([0], dtype=int)

        if num_speakers is not None:
            k = max(1, min(num_speakers, n_samples))
            if k == 1:
                return np.zeros(n_samples, dtype=int)
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            return clustering.fit_predict(embeddings)

        # Automatic speaker count estimation via silhouette score
        max_k = min(self.max_speakers, n_samples - 1)
        if max_k < 2:
            return np.zeros(n_samples, dtype=int)

        best_score = -1.0
        best_k = 1
        best_labels = np.zeros(n_samples, dtype=int)

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            if len(np.unique(labels)) > 1:
                try:
                    score = silhouette_score(embeddings, labels, metric="cosine")
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                except Exception:
                    pass

        # If silhouette is weak, assume 2 speakers for typical multi-talker mixture
        if best_score < 0.15 and n_samples >= 4:
            clustering = AgglomerativeClustering(
                n_clusters=2,
                metric="cosine",
                linkage="average",
            )
            return clustering.fit_predict(embeddings)

        return best_labels

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Diarize mono audio into timestamped speaker segments.
        """
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate must be {self.sample_rate} Hz (got {sample_rate} Hz)."
            )

        audio_mono = np.asarray(audio, dtype=np.float32)
        if audio_mono.ndim != 1:
            raise ValueError("Audio must be 1-dimensional mono array.")

        total_audio_duration = float(len(audio_mono) / self.sample_rate)

        # 1. Voice Activity Detection
        active_regions = self._voice_activity_detection(audio_mono)

        if not active_regions:
            return DiarizationResult(
                segments=[],
                speakers=[],
                speaker_stats={},
                total_audio_duration=round(total_audio_duration, 3),
                total_speech_duration=0.0,
                sample_rate=self.sample_rate,
                num_speakers=0,
            )

        # 2. Extract window embeddings across active speech regions
        window_meta: List[Tuple[float, float]] = []
        window_embeddings: List[np.ndarray] = []

        for region_start, region_end in active_regions:
            region_len = region_end - region_start

            if region_len < self.window_samples:
                chunk = audio_mono[region_start:region_end]
                embedding = self._extract_embedding(chunk)
                window_embeddings.append(embedding)
                window_meta.append((
                    region_start / self.sample_rate,
                    region_end / self.sample_rate,
                ))
            else:
                cur = region_start
                while cur + self.window_samples <= region_end:
                    chunk = audio_mono[cur:cur + self.window_samples]
                    embedding = self._extract_embedding(chunk)
                    window_embeddings.append(embedding)
                    window_meta.append((
                        cur / self.sample_rate,
                        (cur + self.window_samples) / self.sample_rate,
                    ))
                    cur += self.window_hop_samples

                if cur < region_end and (region_end - cur) >= int(0.3 * self.sample_rate):
                    chunk = audio_mono[cur:region_end]
                    embedding = self._extract_embedding(chunk)
                    window_embeddings.append(embedding)
                    window_meta.append((
                        cur / self.sample_rate,
                        region_end / self.sample_rate,
                    ))

        if not window_embeddings:
            return DiarizationResult(
                segments=[],
                speakers=[],
                speaker_stats={},
                total_audio_duration=round(total_audio_duration, 3),
                total_speech_duration=0.0,
                sample_rate=self.sample_rate,
                num_speakers=0,
            )

        # 3. Clustering
        embeddings_arr = np.array(window_embeddings, dtype=np.float32)
        raw_labels = self._cluster_windows(embeddings_arr, num_speakers=num_speakers)

        # 4. Chronological Speaker Mapping:
        # First unique speaker detected gets "Speaker A", second gets "Speaker B", etc.
        cluster_to_name: Dict[int, str] = {}
        speaker_letters = list(string.ascii_uppercase)

        for label in raw_labels:
            if label not in cluster_to_name:
                idx = len(cluster_to_name)
                name = (
                    f"Speaker {speaker_letters[idx]}"
                    if idx < len(speaker_letters)
                    else f"Speaker {idx+1}"
                )
                cluster_to_name[label] = name

        # 5. Build raw segments
        raw_segments: List[SpeakerSegment] = []
        for (w_start, w_end), label in zip(window_meta, raw_labels):
            raw_segments.append(SpeakerSegment(
                speaker=cluster_to_name[label],
                start_time=round(float(w_start), 3),
                end_time=round(float(w_end), 3),
                duration=round(float(w_end - w_start), 3),
                confidence=0.88,
            ))

        # 6. Merge adjacent windows from same speaker with small gap
        merged_segments: List[SpeakerSegment] = []
        for seg in raw_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                if (
                    last_seg.speaker == seg.speaker
                    and (seg.start_time - last_seg.end_time) <= self.merge_gap_threshold
                ):
                    new_end = max(last_seg.end_time, seg.end_time)
                    new_dur = round(new_end - last_seg.start_time, 3)
                    merged_segments[-1] = SpeakerSegment(
                        speaker=last_seg.speaker,
                        start_time=last_seg.start_time,
                        end_time=new_end,
                        duration=new_dur,
                        confidence=last_seg.confidence,
                    )
                else:
                    merged_segments.append(seg)

        # 7. Aggregate Speaker Statistics
        detected_speakers = sorted(list(cluster_to_name.values()))
        speaker_stats: Dict[str, SpeakerStats] = {}

        for spk in detected_speakers:
            spk_segs = [s for s in merged_segments if s.speaker == spk]
            spk_dur = sum(s.duration for s in spk_segs)
            spk_count = len(spk_segs)
            pct = (spk_dur / total_audio_duration * 100.0) if total_audio_duration > 0 else 0.0
            avg_dur = (spk_dur / spk_count) if spk_count > 0 else 0.0
            speaker_stats[spk] = SpeakerStats(
                speaker=spk,
                total_duration=round(spk_dur, 3),
                num_segments=spk_count,
                percentage=round(pct, 1),
                avg_segment_duration=round(avg_dur, 3),
            )

        total_speech = sum(s.duration for s in merged_segments)

        return DiarizationResult(
            segments=merged_segments,
            speakers=detected_speakers,
            speaker_stats=speaker_stats,
            total_audio_duration=round(total_audio_duration, 3),
            total_speech_duration=round(total_speech, 3),
            sample_rate=self.sample_rate,
            num_speakers=len(detected_speakers),
        )
