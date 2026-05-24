#!/usr/bin/env python3
"""Smoke test image generation without launching Streamlit."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import AIImageGenerator, GenerationSettings, MODEL_OPTIONS, read_secret  # noqa: E402


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image and verify that PNG bytes were returned.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Hugging Face live generation. Requires HF_TOKEN or --token.",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Hugging Face token for live mode. Prefer HF_TOKEN to avoid shell history.",
    )
    parser.add_argument(
        "--allow-live-fallback",
        action="store_true",
        help="Allow live failures to fall back to Sketch mode. Keep this off for real provider tests.",
    )
    parser.add_argument(
        "--prompt",
        default="A compact brass robot painting a sunset over a mountain observatory",
        help="Prompt used for the smoke test.",
    )
    parser.add_argument("--style", default="realistic", help="Style key from app.py.")
    parser.add_argument("--quality", default="Fast", help="Quality preset from app.py.")
    parser.add_argument("--model", default="Auto showcase", choices=sorted(MODEL_OPTIONS), help="Model label.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible output.")
    parser.add_argument(
        "--output",
        default="",
        help="Output PNG path. Defaults to outputs/smoke-sketch.png or outputs/smoke-live.png.",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        style=args.style,
        quality=args.quality,
        model=MODEL_OPTIONS[args.model],
        negative_prompt="blurry, low quality, distorted, watermark, text artifacts",
        seed=args.seed,
        batch_size=1,
        demo_mode=not args.live,
        fallback_enabled=bool(args.allow_live_fallback),
        token_source="Smoke test",
    )


def main() -> int:
    args = parse_args()
    token = args.token.strip() or os.environ.get("HF_TOKEN", "").strip() or read_secret("HF_TOKEN")
    if args.live and not token:
        print("FAIL Missing HF_TOKEN. Set HF_TOKEN or pass --token to test live generation.", file=sys.stderr)
        return 2

    output_path = Path(args.output or f"outputs/smoke-{'live' if args.live else 'sketch'}.png")
    settings = build_settings(args)
    result = AIImageGenerator().generate_one(args.prompt, token, settings, args.seed)

    if not result.image_data:
        print(f"FAIL {result.source}: {result.message}", file=sys.stderr)
        if result.technical_message:
            print(f"DETAIL {result.technical_message}", file=sys.stderr)
        return 1
    if not result.image_data.startswith(PNG_SIGNATURE):
        print(f"FAIL Expected PNG bytes, got {len(result.image_data)} bytes from {result.source}.", file=sys.stderr)
        return 1
    if args.live and result.source != "Hugging Face" and not args.allow_live_fallback:
        print(f"FAIL Live test did not return Hugging Face output: {result.source}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.image_data)
    print(f"OK {result.source}: wrote {len(result.image_data)} bytes to {output_path}")
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
