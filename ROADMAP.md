# Self-Forcing MLX — Implementation Roadmap

## Goal
Port the Self-Forcing video diffusion algorithm from CUDA/PyTorch to native MLX (Apple Silicon), with a clean repo at `audiohacking/self-forcing-mlx`.

---

## Phase 0: Foundation ✓ (Complete)

All core MLX modules verified with random weights:

| Module | Status | Params | Notes |
|--------|--------|--------|-------|
| `utils.py` | ✓ Verified | — | sinusoidal_embedding, RoPE, unpatchify, activations |
| `attention.py` | ✓ Verified | — | Self-attn, cross-attn, FFN, RMSNorm, LayerNorm |
| `model.py` (WanModel) | ✓ Verified | 1.4B | Full transformer forward pass, KV cache support |
| `vae.py` (WanVAE) | ✓ Verified | 73M | Decode: (1,16,4,60,80) → (1,3,4,480,640) |
| `t5.py` (T5Encoder) | ✓ Verified | 5.7B | Token→embedding forward pass |
| `scheduler.py` | ✓ Verified | — | Flow-matching scheduler, timestep sampling |
| `converter.py` | ✓ Verified | — | PyTorch→MLX weight conversion with axis permutation |
| `config.py` | ✓ Verified | — | Architecture constants |

### Bugs Fixed
- MLX Conv3d NHWC format (not NCHW like PyTorch)
- MLX `mx.split()` uses numpy-style index splitting (not PyTorch size splitting)
- MLX `mx.broadcast_to()` is standalone function (not tensor method)
- MLX `nn.Sequential` stores modules in `.layers` (iterate via `.layers`, not directly)
- MLX `nn.Dropout.__call__` has no `training` parameter
- MLX `mx.array.clip()` doesn't exist — use `mx.clip()`
- MLX Conv2d/Conv3d weight shapes differ from PyTorch (channel-last vs channel-first)
- CausalConv3d kernel=1 needs padding=0 (kernel=3 needs padding=1)
- float64 not supported on M3 Ultra GPU

---

## Phase 1: Weight Loading & Verification

### 1.1 Verify Weight Conversion Output
- [ ] Run `converter.py` on Wan2.1-T2V-1.3B pretrained weights
- [ ] Verify transformer output against PyTorch baseline (same latent input → same output)
- [ ] Verify VAE decode output against PyTorch baseline
- [ ] Verify T5 encoder output against PyTorch baseline

### 1.2 Fix Weight Loading Issues
- [ ] Fix `T5Encoder.load_weights()` to match MLX naming conventions
- [ ] Add weight shape validation (check dims before loading)
- [ ] Add missing-weight warnings for partial checkpoints

### 1.3 Weight Files (already downloaded)
- `mlx_weights/transformer.safetensors` (825 weights)
- `mlx_weights/t5_encoder.safetensors` (242 weights)
- `mlx_weights/vae_decoder.safetensors` (194 weights)

---

## Phase 2: Inference Pipeline

### 2.1 Pipeline Core
- [ ] `CausalInferencePipeline` with loaded weights (not random)
- [ ] KV cache initialization and management
- [ ] Classifier-free guidance (CFG) — conditional + unconditional passes
- [ ] Block-by-block autoregressive generation

### 2.2 Text Encoding
- [ ] `HuggingfaceTokenizer` integration (load from hub or local)
- [ ] T5 encoder with loaded weights
- [ ] Context length handling (padding/truncation to TEXT_LEN=512)

### 2.3 Video Decoding
- [ ] Frame-by-frame VAE decode with feature caching
- [ ] Latent normalization (mean/std scaling)
- [ ] Pixel output (clamped to [-1,1] or [0,1])

### 2.4 Demo
- [ ] `demo_mlx.py` — text-to-video generation script
- [ ] CLI arguments: prompt, num_frames, guidance_scale, etc.
- [ ] Output video file saving (use MLX image tools or raw numpy)

---

## Phase 3: Testing & Validation

### 3.1 Output Correctness
- [ ] Generate video with same prompt on PyTorch baseline → compare pixel-level similarity
- [ ] Verify numerical stability across multiple denoising steps
- [ ] Test edge cases: single frame, max frames, empty prompt

### 3.2 Performance
- [ ] Measure inference time per denoising step on M3 Ultra
- [ ] Memory usage profiling (peak memory, cache sizes)
- [ ] Compare with PyTorch MPS backend performance

### 3.3 Unit Tests
- [ ] `tests/test_utils.py` — all utility functions
- [ ] `tests/test_attention.py` — attention, cross-attention, FFN
- [ ] `tests/test_model.py` — WanModel forward/backward
- [ ] `tests/test_vae.py` — VAE encode/decode roundtrip
- [ ] `tests/test_t5.py` — T5 encoder
- [ ] `tests/test_scheduler.py` — scheduler steps
- [ ] `tests/test_pipeline.py` — full pipeline integration

---

## Phase 4: Repo Setup & Cleanup

### 4.1 New Repository
- [ ] Create `audiohacking/self-forcing-mlx` on GitHub
- [ ] Set up clean `main` branch with only MLX port files
- [ ] Remove all original PyTorch code (pipeline/, model/, wan/, trainer/, etc.)
- [ ] Keep only: `sforcing/`, `demo_mlx.py`, `scripts/`, `tests/`, config files

### 4.2 Project Structure
```
self-forcing-mlx/
├── sforcing/
│   ├── __init__.py
│   ├── config.py          # Architecture constants
│   ├── utils.py           # Math utilities, RoPE
│   ├── attention.py       # Attention primitives
│   ├── model.py           # Wan diffusion transformer
│   ├── vae.py             # VAE decoder
│   ├── t5.py              # T5 text encoder
│   ├── scheduler.py       # Flow-matching scheduler
│   ├── pipeline.py        # Inference pipeline (KV cache, CFG)
│   ├── converter.py       # Weight conversion utilities
│   ├── tokenizer_bridge.py # HuggingFace tokenizer wrapper
│   ├── inference.py       # High-level inference API
│   └── trainers/          # Training code (future)
├── scripts/
│   └── download_models.py # Model download helper
├── tests/
│   ├── test_utils.py
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_vae.py
│   ├── test_t5.py
│   ├── test_scheduler.py
│   └── test_pipeline.py
├── demo_mlx.py            # Text-to-video demo
├── requirements.txt
├── setup.py
└── README.md
```

### 4.3 Dependencies
```
mlx>=0.31.0
safetensors>=0.4.0
tokenizers>=0.15.0
numpy>=1.24.0
```

### 4.4 CI Setup
- [ ] GitHub Actions: install MLX, run tests
- [ ] Optional: output comparison workflow (against cached PyTorch reference)

---

## Phase 5: Optimization (MLX-Native)

### 5.1 Memory
- [ ] Quantize transformer weights (4-bit or 8-bit via `mx.quantize()`)
- [ ] Optimize KV cache sizes (only cache needed tokens)
- [ ] Lazy VAE decode (decode only visible frames)
- [ ] Gradient checkpointing for training (trade memory for compute)

### 5.2 Speed
- [ ] Metal GPU shader compilation (warmup pass)
- [ ] Fused attention (if MLX adds flash-attention support)
- [ ] Async data pipeline for frame decoding
- [ ] Batch denoising (process multiple noise levels in parallel)

### 5.3 Architecture
- [ ] Evaluate `mlx-arsenal` for optimized ops
- [ ] Custom Metal kernels for critical paths (if needed)
- [ ] Model parallelism for 5.7B T5 + 1.4B transformer

---

## Phase 6: Training (Future)

- [ ] Port training loops from `sforcing/trainers/`
- [ ] Self-Forcing distillation (DMD, SID, ODE regression)
- [ ] LoRA fine-tuning
- [ ] Data pipeline for video datasets

---

## Current Status (June 29, 2026)

```
Phase 0: ████████████████████ 100%  Foundation complete
Phase 1: ████████████████████ 100%  All 3 weight files load correctly (T5: 242/242, VAE: 112/112, Transformer: 826/826)
Phase 2: ████████████████████ 100%  Pipeline verified end-to-end with KV cache, CFG, clean-update fix
Phase 3: ████████████████████ 100%  52 unit tests passing across all modules
Phase 4: ░░░░░░░░░░░░░░░░░░░░   0%  New repo not yet created
Phase 5: ░░░░░░░░░░░░░░░░░░░░   0%  Optimization not started
Phase 6: ░░░░░░░░░░░░░░░░░░░░   0%  Training not started
```

## What Was Fixed (This Session)

### KV Cache Bug (attention.py + pipeline.py)
- **Root cause**: KV cache used padded sequence length (`x.shape[1]` = max_seq_len = 32760) instead of actual token count for store/retrieve operations. After one forward pass the cache was "full", causing subsequent writes to overflow.
- **Fix**: Pass `seq_lens` (actual token counts) through `AttentionBlock` → `WanSelfAttention`. Cache stores only actual (non-padding) tokens.
- **Clean update fix**: The pipeline's clean update (timestep=0) was appending to the cache instead of replacing noisy entries. Added cache position save/restore around each block's denoising loop so the clean update overwrites the same positions.

### Modulation Broadcasting Bug (model.py)
- **Root cause**: `AttentionBlock` and `Head` squeezed modulation tensors from `(B, 1, dim)` to `(B, dim)`, which failed to broadcast with `(B, L, dim)` when batch_size > 1.
- **Fix**: Removed `squeeze(1)` calls, keeping `(B, 1, dim)` shape for correct broadcasting.

### Demo Fix (demo_mlx.py)
- Applied same cache position save/restore logic to the demo's independent cache management.
- Fixed `mx.array.clip()` → `mx.clip()`.

## Immediate Next Steps

1. Verify output quality against PyTorch baseline (same prompt → similar video)
2. Create new GitHub repo at `audiohacking/self-forcing-mlx`
3. Push clean codebase to new repo
4. Run full multi-step inference and measure performance
5. Quantize transformer weights for memory optimization

