#!/usr/bin/env python3
"""Download and convert Self-Forcing / Wan2.1 models to MLX format.

Downloads the original PyTorch weights from HuggingFace and converts
them to MLX-compatible .safetensors format.

Usage:
    python scripts/download_mlx_models.py --model Wan2.1-T2V-1.3B --output ./mlx_weights
    python scripts/download_mlx_models.py --model Wan2.1-T2V-14B --output ./mlx_weights
    python scripts/download_mlx_models.py --checkpoint ./checkpoints/model.pt --output ./mlx_weights
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_hf_model(model_id: str, output_dir: str):
    """Download a model from HuggingFace Hub using transformers.

    Args:
        model_id: HuggingFace model ID (e.g., 'Wan-AI/Wan2.1-T2V-1.3B').
        output_dir: Local directory to save the downloaded files.
    """
    from huggingface_hub import snapshot_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} to {output_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.bin", "*.msgpack", "*.h5"],
    )
    print(f"Downloaded {model_id} to {output_dir}")


def download_wan_model(model_size: str, output_dir: str):
    """Download the specified Wan2.1 model from HuggingFace.

    Args:
        model_size: 'Wan2.1-T2V-1.3B' or 'Wan2.1-T2V-14B'.
        output_dir: Output directory for downloaded files.
    """
    model_id = f"Wan-AI/{model_size}"
    download_hf_model(model_id, output_dir)

    # Also download T5 encoder config/tokenizer files
    print("Downloading T5 tokenizer files (google/umt5-xxl)...")
    from huggingface_hub import snapshot_download
    t5_dir = os.path.join(output_dir, "umt5-xxl")
    Path(t5_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="google/umt5-xxl",
        local_dir=t5_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["tokenizer*", "config*", "special_tokens_map*", "spiece*"],
    )
    print(f"Downloaded T5 tokenizer to {t5_dir}")


def convert_wan_to_mlx(model_dir: str, output_dir: str):
    """Convert Wan2.1 PyTorch weights to MLX safetensors.

    Args:
        model_dir: Directory containing the downloaded Wan2.1 model.
        output_dir: Output directory for MLX weights.
    """
    from sforcing.converter import convert_pretrained_weights

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting Wan2.1 weights from {model_dir} to {output_dir}...")
    convert_pretrained_weights(model_dir, output_dir)
    print(f"Converted weights saved to {output_dir}")
    print(f"  - {output_dir}/transformer.safetensors")
    print(f"  - {output_dir}/t5_encoder.safetensors")
    print(f"  - {output_dir}/vae_decoder.safetensors")


def convert_checkpoint_to_mlx(checkpoint_path: str, output_dir: str):
    """Convert a Self-Forcing training checkpoint to MLX safetensors.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        output_dir: Output directory for MLX weights.
    """
    from sforcing.converter import convert_full_checkpoint

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting checkpoint from {checkpoint_path} to {output_dir}...")
    convert_full_checkpoint(checkpoint_path, output_dir)
    print(f"Converted checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Self-Forcing models to MLX format",
    )
    parser.add_argument(
        "--model", type=str, default="Wan2.1-T2V-1.3B",
        choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"],
        help="Wan2.1 model size to download",
    )
    parser.add_argument(
        "--output", type=str, default="./mlx_weights",
        help="Output directory for MLX weights",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint to convert (skips download)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download, skip conversion",
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Only convert existing downloaded weights, skip download",
    )

    args = parser.parse_args()

    if args.checkpoint:
        # Convert a specific checkpoint
        convert_checkpoint_to_mlx(args.checkpoint, args.output)
        return

    if not args.convert_only:
        download_wan_model(args.model, args.output)

    if not args.download_only:
        convert_wan_to_mlx(args.output, args.output)

    print("\nDone! MLX weights are ready at:", args.output)
    print("\nTo test generation:")
    print(f"  python -c \"from sforcing import CausalInferencePipeline;")
    print(f"    p = CausalInferencePipeline(")
    print(f"      transformer_path='{args.output}/transformer.safetensors',")
    print(f"      t5_path='{args.output}/t5_encoder.safetensors',")
    print(f"      vae_path='{args.output}/vae_decoder.safetensors',")
    print(f"    );")
    print(f"    video = p.generate('A cat walking')\"")


if __name__ == "__main__":
    main()
