"""Small command-line interface; no training or account is required."""

import argparse
from pathlib import Path

import torch

from .api import KokoroCloner


def main():
    parser = argparse.ArgumentParser(
        description="Clone an English reference, or use original Kokoro with --adapter off."
    )
    parser.add_argument("--adapter", choices=("on", "off"), default="on")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--text")
    target.add_argument("--text-file", type=Path)
    parser.add_argument("--reference", type=Path)
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument("--reference-text")
    transcript.add_argument("--reference-text-file", type=Path)
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Original English voice ID, used with --adapter off",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional offline folder containing the four release model files",
    )
    parser.add_argument(
        "--base-dir", type=Path, help="Optional offline original Kokoro folder"
    )
    parser.add_argument("--wavlm-dir", type=Path, help="Optional offline WavLM folder")
    parser.add_argument(
        "--offline", action="store_true", help="Use cached/downloaded files only"
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    if args.adapter == "on" and (
        not args.reference or not (args.reference_text or args.reference_text_file)
    ):
        parser.error(
            "Cloning needs --reference and --reference-text or --reference-text-file"
        )
    if args.adapter == "off" and (
        args.reference or args.reference_text or args.reference_text_file
    ):
        parser.error("Adapter off uses --voice; omit reference arguments")
    torch.set_num_threads(args.threads)
    try:
        text = (
            args.text
            if args.text is not None
            else args.text_file.read_text(encoding="utf-8")
        )
        model = KokoroCloner(
            args.model_dir,
            base_dir=args.base_dir,
            wavlm_dir=args.wavlm_dir,
            device=args.device,
            local_files_only=args.offline,
        )
        model.set_adapter_enabled(args.adapter == "on")
        kwargs = dict(seed_value=args.seed)
        if args.adapter == "on":
            transcript = (
                args.reference_text
                if args.reference_text is not None
                else args.reference_text_file.read_text(encoding="utf-8")
            )
            print("Reading reference and creating voice conditioning...", flush=True)
            kwargs["reference"] = model.enroll(args.reference, transcript)
        else:
            kwargs["voice"] = args.voice
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(text, args.output, **kwargs)
        print(f"Saved {args.output} (24 kHz mono WAV; adapter {args.adapter})")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
