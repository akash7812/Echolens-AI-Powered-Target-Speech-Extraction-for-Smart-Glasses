"""
Echolens Command-Line Interface (CLI)
=====================================
Unified interface for CHiME-10 Task 3:
- Track 1: Wearer Own-Speech Enhancement
- Track 2: Interlocutor Target Speaker Extraction
- Web Application Dashboard Server
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf

from echolens.audio.ingestion import ingest_audio
from echolens.pipeline.track1 import Track1Pipeline
from echolens.pipeline.track2 import Track2Pipeline, DEFAULT_CHECKPOINT_PATH
from echolens.diarization.offline_diarizer import SpectralAHCDiarizer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echolens",
        description="Echolens: AI-Powered Target Speech Extraction for Smart Glasses (CHiME-10 Task 3)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Server subcommand
    default_port = int(os.environ.get("PORT", 8000))
    server_parser = subparsers.add_parser("server", help="Launch the Echolens interactive web dashboard")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0, binds to all interfaces)")
    server_parser.add_argument("--port", type=int, default=default_port, help=f"Port number (default: $PORT or {default_port})")
    server_parser.add_argument("--storage-dir", type=str, default=None, help="Directory for audio cache and output")

    # Track 1 subcommand
    t1_parser = subparsers.add_parser("track1", help="Track 1: Wearer Own-Speech Enhancement")
    t1_parser.add_argument("--mixture", type=str, required=True, help="Path to input mixture audio WAV")
    t1_parser.add_argument("--reference", type=str, required=True, help="Path to wearer voice reference WAV")
    t1_parser.add_argument("--target", type=str, default=None, help="Optional clean target speech WAV for ground-truth evaluation")
    t1_parser.add_argument("--output", type=str, default="enhanced_own_speech.wav", help="Output WAV path")
    t1_parser.add_argument("--json", action="store_true", help="Output result summary as JSON")

    # Track 2 subcommand
    t2_parser = subparsers.add_parser("track2", help="Track 2: Interlocutor Target Speaker Extraction")
    t2_parser.add_argument("--mixture", type=str, required=True, help="Path to input mixture audio WAV")
    t2_group = t2_parser.add_mutually_exclusive_group(required=True)
    t2_group.add_argument("--enrollment", type=str, help="Path to pre-extracted enrollment audio WAV")
    t2_group.add_argument("--speaker", type=str, help="Speaker ID to extract via diarization (e.g. 'Speaker A')")
    t2_parser.add_argument("--checkpoint", type=str, default=None, help="Path to PS4 model checkpoint .pt")
    t2_parser.add_argument("--target", type=str, default=None, help="Optional clean target speech WAV for ground-truth evaluation")
    t2_parser.add_argument("--output", type=str, default="extracted_target_speaker.wav", help="Output WAV path")
    t2_parser.add_argument("--json", action="store_true", help="Output result summary as JSON")

    # Info subcommand
    subparsers.add_parser("info", help="Display Echolens system diagnostics and environment info")

    return parser


def cmd_info() -> int:
    import torch

    print("================================================================")
    print(" Echolens System Diagnostics (CHiME-10 Task 3)")
    print("================================================================")
    print(f" Python Version : {sys.version.split()[0]}")
    print(f" PyTorch Version: {torch.__version__}")
    print(f" CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" Device Name    : {torch.cuda.get_device_name(0)}")
    else:
        print(" Device Name    : CPU (Fallback verified)")
    
    ckpt_path = Path(DEFAULT_CHECKPOINT_PATH)
    print(f" PS4 Checkpoint : {ckpt_path}")
    print(f" Checkpoint Exists: {ckpt_path.exists()}")
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f" Checkpoint Size: {size_mb:.2f} MB")
    
    print("\n Supported Tracks:")
    print("  - Track 1: Wearer Own-Speech Enhancement (Biometric Voice Conditioning)")
    print("  - Track 2: Interlocutor Target Speaker Extraction (PS4 Deep Conv-TasNet)")
    print("================================================================")
    return 0


def cmd_server(args) -> int:
    from echolens.app.server import create_server
    import socket
    storage_dir = Path(args.storage_dir) if args.storage_dir else None

    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print(f"================================================================", flush=True)
    print(f" 🚀 Echolens Web Dashboard Deployed & Active", flush=True)
    print(f"    Local Access   : http://localhost:{args.port}", flush=True)
    if args.host in ("0.0.0.0", ""):
        print(f"    Network Access : http://{local_ip}:{args.port}", flush=True)
    else:
        print(f"    Binding Host   : http://{args.host}:{args.port}", flush=True)
    print(f"    Press Ctrl+C to stop the server.", flush=True)
    print(f"================================================================", flush=True)
    server = create_server(host=args.host, port=args.port, storage_dir=storage_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Echolens Web Dashboard.", flush=True)
        server.shutdown()
        server.server_close()
    return 0


def cmd_track1(args) -> int:
    mix_ingest = ingest_audio(args.mixture)
    ref_ingest = ingest_audio(args.reference)
    target_audio = ingest_audio(args.target).audio if args.target else None

    pipeline = Track1Pipeline()
    result = pipeline.enhance(
        mixture=mix_ingest.audio,
        wearer_reference=ref_ingest.audio,
        clean_target=target_audio,
        output_path=args.output,
    )

    if args.json:
        data = result.to_dict()
        data["success"] = True
        print(json.dumps(data, indent=2))
    else:
        print("================================================================")
        print(" Track 1: Wearer Own-Speech Enhancement Complete")
        print("================================================================")
        print(f" Enhanced Output  : {result.output_path}")
        print(f" Duration         : {result.duration:.2f} s")
        print(f" Noise Reduction  : {result.noise_reduction_db:.2f} dB")
        print(f" Execution Time   : {result.execution_time:.3f} s")
        if result.metrics:
            print(" Evaluation Metrics (Ground-Truth Target Reference):")
            print(f"   Mixture SI-SDR : {result.metrics.get('mixture_si_sdr', 'N/A')} dB")
            print(f"   Enhanced SI-SDR: {result.metrics.get('enhanced_si_sdr', 'N/A')} dB")
            print(f"   SI-SDR Imprv   : {result.metrics.get('si_sdr_improvement', 'N/A')} dB")
        print("================================================================")
    return 0


def cmd_track2(args) -> int:
    mix_ingest = ingest_audio(args.mixture)
    target_audio = ingest_audio(args.target).audio if args.target else None

    ckpt = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT_PATH
    pipeline = Track2Pipeline(checkpoint_path=ckpt)

    if args.enrollment:
        enroll_ingest = ingest_audio(args.enrollment)
        result = pipeline.extract(
            mixture=mix_ingest.audio,
            enrollment_audio=enroll_ingest.audio,
            clean_target=target_audio,
            output_path=args.output,
        )
    else:
        # Run diarization first
        diarizer = SpectralAHCDiarizer()
        diar_res = diarizer.diarize(mix_ingest.audio, sample_rate=mix_ingest.sample_rate)
        result = pipeline.extract(
            mixture=mix_ingest.audio,
            diarization_result=diar_res,
            selected_speaker=args.speaker,
            clean_target=target_audio,
            output_path=args.output,
        )

    if args.json:
        data = result.to_dict()
        data["success"] = True
        print(json.dumps(data, indent=2))
    else:
        print("================================================================")
        print(" Track 2: Interlocutor Target Speaker Extraction Complete")
        print("================================================================")
        print(f" Extracted Output : {result.output_path}")
        print(f" Duration         : {result.duration:.2f} s")
        print(f" Selected Speaker : {result.selected_speaker or 'Direct Enrollment'}")
        print(f" Execution Time   : {result.execution_time:.3f} s")
        if result.metrics:
            print(" Evaluation Metrics (Ground-Truth Target Reference):")
            print(f"   Mixture SI-SDR : {result.metrics.get('mixture_si_sdr', 'N/A')} dB")
            print(f"   Extracted SI-SDR: {result.metrics.get('extracted_si_sdr', 'N/A')} dB")
            print(f"   SI-SDR Imprv   : {result.metrics.get('si_sdr_improvement', 'N/A')} dB")
        print("================================================================")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    if args.command == "info":
        return cmd_info()
    elif args.command == "server":
        return cmd_server(args)
    elif args.command == "track1":
        return cmd_track1(args)
    elif args.command == "track2":
        return cmd_track2(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
