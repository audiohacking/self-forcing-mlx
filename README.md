# Self-Forcing MLX

Native [MLX](https://github.com/ml-explore/mlx) port of [Self-Forcing](https://self-forcing.github.io) — the autoregressive video diffusion model, optimized for Apple Silicon.

## Overview

This is a clean, standalone MLX implementation of the Wan2.1-T2V-1.3B video diffusion backbone with Self-Forcing inference. It runs entirely on Apple Silicon via Metal (no CUDA, no PyTorch).

### Features

- **KV-cache-based autoregressive inference** — block-by-block video generation
- **Classifier-free guidance (CFG)** — conditional + unconditional passes
- **Flow-matching scheduler** — configurable denoising step schedules
- **Image-to-video** — via initial latent conditioning
- **Web demo** — real-time frame streaming via Flask + SocketIO

## Requirements

- Apple Silicon Mac (M-series)
- macOS with Metal support
- Python 3.10+

## Installation

```bash
python -m venv mlx-venv
source mlx-venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Download Weights

```bash
python scripts/download_mlx_models.py --output ./mlx_weights
```

Or manually place these files in `./mlx_weights/`:
- `transformer.safetensors` (1.3B params)
- `t5_encoder.safetensors` (5.7B params)
- `vae_decoder.safetensors` (73M params)

## Usage

### CLI Demo (Web UI)

```bash
python demo_mlx.py --port 5001
```

### Programmatic

```python
from sforcing.pipeline import CausalInferencePipeline

pipeline = CausalInferencePipeline(
    transformer_path="mlx_weights/transformer.safetensors",
    t5_path="mlx_weights/t5_encoder.safetensors",
    vae_path="mlx_weights/vae_decoder.safetensors",
)

video = pipeline.generate("a cat walking in the park")
```

## Project Structure

```
sforcing/
├── __init__.py
├── config.py          # Architecture constants
├── utils.py           # Math utilities, RoPE
├── attention.py       # Self-attention, cross-attention, FFN
├── model.py           # Wan diffusion transformer
├── vae.py             # VAE decoder
├── t5.py              # T5 text encoder
├── scheduler.py       # Flow-matching scheduler
├── pipeline.py        # Inference pipeline (KV cache, CFG)
├── converter.py       # Weight conversion utilities
├── tokenizer_bridge.py
├── inference.py       # High-level inference API
├── data.py            # Data utilities
└── trainers/          # Training strategies (DMD, SID, GAN, etc.)
```

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
