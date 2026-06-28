"""PyTorch state_dict → .safetensors weight converter.

Converts PyTorch checkpoint files (from the original Wan/Self-Forcing repo)
into MLX-compatible .safetensors files for direct loading.
"""
import os
import argparse

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM, TEXT_DIM, TEXT_LEN,
    IN_DIM, OUT_DIM, PATCH_SIZE, EPS,
)


def convert_state_dict(state_dict: dict) -> dict:
    """Convert PyTorch state_dict to MLX-compatible naming convention.

    Strips 'module.' prefix from DDP/FSDP checkpoints and renames
    PyTorch parameter names to MLX conventions.

    Args:
        state_dict: PyTorch state dict.

    Returns:
        MLX-compatible state dict with numpy arrays.
    """
    import torch

    converted = {}
    for key, value in state_dict.items():
        # Strip DDP/FSDP prefix
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("generator."):
            key = key[10:]

        # Convert to numpy, bf16 → float32 for safe storage
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu()
            if v.dtype == torch.bfloat16:
                v = v.float()
            np_arr = v.numpy()
            converted[key] = np_arr
        else:
            converted[key] = value

    return converted


def convert_transformer(
    torch_path: str,
    mlx_path: str,
):
    """Convert transformer weights only.

    Args:
        torch_path: Path to .pt checkpoint file.
        mlx_path: Output .safetensors path.
    """
    import torch
    from safetensors.numpy import save_file

    print(f"Loading transformer from {torch_path}")
    state_dict = torch.load(torch_path, map_location="cpu", weights_only=False)

    # Extract the model state dict (could be under 'state_dict' key)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    converted = convert_state_dict(state_dict)
    save_file(converted, mlx_path)
    print(f"Saved transformer to {mlx_path}")


def convert_full_checkpoint(
    torch_path: str,
    output_dir: str,
):
    """Convert full checkpoint (transformer + T5 + VAE) into separate files.

    Uses priority-based key matching to avoid overlap between components.
    VAE weights matched first (most specific), then transformer, then T5.

    Args:
        torch_path: Path to full checkpoint.
        output_dir: Directory for output .safetensors files.
    """
    import torch
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)

    state_dict = torch.load(torch_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    converted = convert_state_dict(state_dict)

    # Priority-based key matching (most specific first)
    # VAE decoder keys: contain "decoder." or "Residual" or "Resample" or start with "model.decoder"
    vae_keys = [
        k for k in converted
        if k.startswith("decoder.") or k.startswith("model.decoder.")
        or "Residual" in k or "Resample" in k
        or k.startswith("conv1.") or k.startswith("conv2.")
    ]

    # Transformer keys: blocks, patch_embedding, head, time/text embedding
    transformer_keys = [
        k for k in converted if k not in vae_keys and (
            "blocks" in k or k.startswith("patch_embedding")
            or k.startswith("head.") or k.startswith("time_embedding")
            or k.startswith("text_embedding") or k.startswith("time_projection")
        )
    ]

    # T5 encoder keys: everything else with token_embedding, pos_embedding,
    # encoder prefix, or remaining block/norm keys
    t5_keys = [
        k for k in converted if k not in vae_keys and k not in transformer_keys and (
            "token_embedding" in k or "pos_embedding" in k
            or k.startswith("encoder.") or k.startswith("shared.")
            or k.startswith("norm.") or k.startswith("blocks.")
        )
    ]

    # Save transformer
    transformer_dict = {k: converted[k] for k in transformer_keys if k in converted}
    if transformer_dict:
        save_file(transformer_dict, os.path.join(output_dir, "transformer.safetensors"))
        print(f"Saved transformer ({len(transformer_dict)} weights)")

    # Save T5 encoder
    t5_dict = {k: converted[k] for k in t5_keys if k in converted}
    if t5_dict:
        save_file(t5_dict, os.path.join(output_dir, "t5_encoder.safetensors"))
        print(f"Saved T5 encoder ({len(t5_dict)} weights)")

    # Save VAE
    vae_dict = {k: converted[k] for k in vae_keys if k in converted}
    if vae_dict:
        save_file(vae_dict, os.path.join(output_dir, "vae_decoder.safetensors"))
        print(f"Saved VAE decoder ({len(vae_dict)} weights)")

    # Warn about unmatched keys
    matched = set(vae_keys) | set(transformer_keys) | set(t5_keys)
    unmatched = set(converted.keys()) - matched
    if unmatched:
        print(f"Warning: {len(unmatched)} unmatched keys not saved to any file")


def convert_pretrained_weights(
    wan_model_dir: str,
    output_dir: str,
):
    """Convert weights from Wan2.1 pretrained model directory.

    Reads the individual .pth files from the official Wan2.1 checkpoint
    directory and outputs separate .safetensors files.

    Args:
        wan_model_dir: Path to extracted Wan2.1 model directory (e.g. wan_models/Wan2.1-T2V-1.3B/).
        output_dir: Output directory for .safetensors files.
    """
    os.makedirs(output_dir, exist_ok=True)

    import glob
    import torch
    from safetensors.numpy import save_file

    # Convert T5 encoder
    t5_path = os.path.join(wan_model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    if os.path.exists(t5_path):
        print(f"Loading T5 from {t5_path}")
        state_dict = torch.load(t5_path, map_location="cpu", weights_only=False)
        if not isinstance(state_dict, dict):
            # Sometimes it's a full model checkpoint
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            else:
                raise ValueError(f"Unexpected T5 checkpoint format: {type(state_dict)}")
        converted = convert_state_dict(state_dict)
        save_file(converted, os.path.join(output_dir, "t5_encoder.safetensors"))
        print(f"Saved T5 encoder ({len(converted)} weights)")

    # Convert VAE
    vae_path = os.path.join(wan_model_dir, "Wan2.1_VAE.pth")
    if os.path.exists(vae_path):
        print(f"Loading VAE from {vae_path}")
        state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)
        converted = convert_state_dict(state_dict)
        save_file(converted, os.path.join(output_dir, "vae_decoder.safetensors"))
        print(f"Saved VAE decoder ({len(converted)} weights)")

    # Convert transformer (from HuggingFace safetensors format)
    transformer_path = os.path.join(wan_model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(transformer_path):
        print(f"Loading transformer from {transformer_path}")
        from safetensors import safe_open
        params = {}
        with safe_open(transformer_path, framework="numpy") as f:
            for key in f.keys():
                params[key] = np.array(f.get_tensor(key))

        # Map HuggingFace/PyTorch names to MLX names.
        # MLX uses dot notation for list indices (blocks.0), not bracket notation.
        # MLX nn.Sequential submodules are accessed via .layers.N, not .N.
        mlx_params = {}
        import re
        for key, value in params.items():
            ml_key = key
            # blocks.N. -> blocks.N.  (same — MLX uses dot notation for list indices)
            # ffn.N. -> ffn.fc1. / ffn.fc2.  (Sequential -> named fields)
            ml_key = re.sub(r'ffn\.0\.', 'ffn.fc1.', ml_key)
            ml_key = re.sub(r'ffn\.2\.', 'ffn.fc2.', ml_key)
            # Sequential layers: text_embedding.N. -> text_embedding.layers.N.
            ml_key = re.sub(r'text_embedding\.(\d+)\.', r'text_embedding.layers.\1.', ml_key)
            ml_key = re.sub(r'time_embedding\.(\d+)\.', r'time_embedding.layers.\1.', ml_key)
            ml_key = re.sub(r'time_projection\.(\d+)\.', r'time_projection.layers.\1.', ml_key)
            # Handle patch_embedding.weight shape: PyTorch Conv3d is (out, in, D, H, W)
            # but MLX Conv3d is (out, D, H, W, in). Need to permute axes.
            if ml_key == 'patch_embedding.weight':
                # Permute (out, in, D, H, W) -> (out, D, H, W, in)
                value = value.transpose(0, 2, 3, 4, 1)
            mlx_params[ml_key] = value

        save_file(mlx_params, os.path.join(output_dir, "transformer.safetensors"))
        print(f"Saved transformer ({len(mlx_params)} weights)")

    print("Done converting pretrained weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to MLX safetensors")
    subparsers = parser.add_subparsers(dest="command")

    p_full = subparsers.add_parser("full", help="Convert a full checkpoint")
    p_full.add_argument("torch_path", help="Path to .pt checkpoint")
    p_full.add_argument("output_dir", help="Output directory")

    p_pretrained = subparsers.add_parser("pretrained", help="Convert Wan2.1 pretrained files")
    p_pretrained.add_argument("wan_model_dir", help="Wan2.1 model directory")
    p_pretrained.add_argument("output_dir", help="Output directory")

    p_transformer = subparsers.add_parser("transformer", help="Convert transformer only")
    p_transformer.add_argument("torch_path", help="Path to transformer .pt")
    p_transformer.add_argument("mlx_path", help="Output .safetensors path")

    args = parser.parse_args()

    if args.command == "full":
        convert_full_checkpoint(args.torch_path, args.output_dir)
    elif args.command == "pretrained":
        convert_pretrained_weights(args.wan_model_dir, args.output_dir)
    elif args.command == "transformer":
        convert_transformer(args.torch_path, args.mlx_path)
    else:
        parser.print_help()