"""VAE decoder ported from PyTorch to MLX.

Ported from wan/modules/vae.py.
Only the Decoder3d is needed for text-to-video inference.
Supports frame-by-frame decoding for memory efficiency.
"""
import mlx.core as mx
import mlx.nn as nn


class CausalConv3d(nn.Module):
    """Causal 3D convolution with temporal-only padding.

    Only the temporal dimension is causally padded (past frames only).
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        if isinstance(padding, tuple):
            self.pad_t = padding[0]
            self.pad_h = padding[1] if len(padding) > 1 else padding[0]
        else:
            self.pad_t = padding
            self.pad_h = padding
        self.out_channels = out_channels

        # Conv3d with no padding — we handle padding manually for causality
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # manual padding for causal
        )

    def __call__(self, x, cache_x=None):
        """Forward pass with causal temporal padding.

        Args:
            x: Input tensor, shape (B, C, T, H, W) — NCHW format.
            cache_x: Cached past frames for temporal causality, or None for first frame.

        Returns:
            Output tensor, shape (B, C_out, T_out, H_out, W_out) — NCHW format.
        """
        # MLX Conv3d expects NHWC format (B, T, H, W, C), not NCHW (B, C, T, H, W)
        x = x.transpose(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)

        if cache_x is not None:
            cache_x = cache_x.astype(x.dtype)
            cache_x = cache_x.transpose(0, 2, 3, 4, 1)
            x = mx.concatenate([cache_x, x], axis=2)

        # Causal padding for temporal, symmetric for spatial
        x = self._pad_causal(x)

        out = self.conv(x)  # (B, T', H', W', C_out)

        # Transpose back to NCHW: (B, T', H', W', C_out) -> (B, C_out, T', H', W')
        out = out.transpose(0, 4, 1, 2, 3)
        return out

    def _pad_causal(self, x):
        """Causal padding ensuring temporal dimension is at least kernel_t.

        Edge-pads (replicates first frame) when temporal context is insufficient
        for the convolution kernel, regardless of cache state.
        """
        b, t, h, w, c = x.shape
        kernel_t = self.conv.weight.shape[1]
        # Spatial padding (symmetric)
        if self.pad_h > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.pad_h, self.pad_h),
                           (self.pad_h, self.pad_h), (0, 0)])
        # Temporal: ensure t >= kernel_t by edge-padding if needed
        if t < kernel_t:
            needed = kernel_t - t
            first = x[:, :1, :, :, :]
            pad_frames = mx.tile(first, (1, needed, 1, 1, 1))
            x = mx.concatenate([pad_frames, x], axis=1)
        return x


class RMSNorm(nn.Module):
    """VAE-specific RMS normalization with learnable scale."""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        self.dim = dim
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = mx.ones((dim,))
        self.bias = mx.zeros((dim,)) if bias else None

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor. If channel_first: (B, C, ...), else (..., C).

        Returns:
            Normalized tensor.
        """
        if self.channel_first:
            # RMS over spatial dims: keep C dim, normalize over rest
            rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=tuple(range(2, x.ndim)), keepdims=True) + 1e-6)
            x_norm = (x.astype(mx.float32) / rms).astype(x.dtype)
        else:
            rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
            x_norm = (x.astype(mx.float32) / rms).astype(x.dtype)

        # Reshape gamma/bias to broadcast with input
        if self.channel_first:
            # (B, C, ...) -> gamma: (C, 1, 1, ...)
            gamma = self.gamma.reshape(-1, *([1] * (x_norm.ndim - 2)))
        else:
            # (..., C) -> gamma: (C,) broadcasts naturally
            gamma = self.gamma
        result = x_norm * self.scale * gamma
        if self.bias is not None:
            bias = self.bias.reshape(gamma.shape) if self.channel_first else self.bias
            result = result + bias
        return result


class ResidualBlock(nn.Module):
    """Residual block with two causal convolutions and RMSNorm + SiLU."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1, padding=0) if in_dim != out_dim else None

    def __call__(self, x, feat_cache=None, feat_idx=None):
        shortcut = self.shortcut(x) if self.shortcut else x
        h = x
        for layer in self.residual.layers:
            if isinstance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                cache_key = feat_idx[0]
                cache_x = h[:, :, -2:, :, :]
                if cache_x.shape[2] < 2 and feat_cache.get(cache_key) is not None:
                    prev = feat_cache[cache_key][:, :, -1:, :, :]
                    cache_x = mx.concatenate([prev, cache_x], axis=2)
                h = layer(h, feat_cache.get(cache_key))
                feat_cache[cache_key] = cache_x
                feat_idx[0] += 1
            else:
                h = layer(h)
        return shortcut + h


class AttentionBlock(nn.Module):
    """Single-head self-attention on spatial dimensions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim, channel_first=False)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor, shape (B, C, T, H, W) — NCHW format.

        Returns:
            Attention output, same format.
        """
        b, c, t, h, w = x.shape
        identity = x

        # MLX Conv2d expects NHWC (B, H, W, C), not NCHW (B, C, H, W)
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W) -> (B*T, H, W, C)
        x_2d = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
        x_2d = self.norm(x_2d)

        # Compute q, k, v
        qkv = self.to_qkv(x_2d)  # (B*T, H, W, 3*C)

        # Split into q, k, v along channel dim (last axis in NHWC)
        q, k, v = qkv.split(3, axis=-1)  # each: (B*T, H, W, C)

        # Reshape for attention: (B*T, H, W, C) -> (B*T, 1, H*W, C)
        c_dim = q.shape[-1]
        q = q.reshape(b * t, h * w, c_dim)[:, None, :, :]  # (B*T, 1, H*W, C)
        k = k.reshape(b * t, h * w, c_dim)[:, None, :, :]  # (B*T, 1, H*W, C)
        v = v.reshape(b * t, h * w, c_dim)[:, None, :, :]  # (B*T, 1, H*W, C)

        # Self-attention
        scale = q.shape[-1] ** -0.5
        attn = mx.matmul(q * scale, k.transpose(0, 1, 3, 2))  # (B*T, 1, H*W, H*W)
        attn = mx.softmax(attn, axis=-1)
        x_out = mx.matmul(attn, v)  # (B*T, 1, H*W, C)
        x_out = x_out.squeeze(1).reshape(b * t, h, w, c_dim)

        x_out = self.proj(x_out)  # (B*T, H, W, C)

        # Reshape back: (B*T, H, W, C) -> (B, T, H, W, C) -> (B, C, T, H, W)
        x_out = x_out.reshape(b, t, h, w, -1).transpose(0, 4, 1, 2, 3)
        return identity + x_out


class Resample(nn.Module):
    """Up/downsample module with temporal causality."""

    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = None

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if self.mode.startswith('upsample') or self.mode.startswith('downsample'):
            b, c, t, h, w = x.shape

            if self.mode.endswith('3d') and 'upsample' in self.mode:
                if feat_cache is not None:
                    idx = feat_idx[0] if feat_idx else 0
                    if feat_cache.get(idx) is not None:
                        cache_x = x[:, :, -2:, :, :]
                        if cache_x.shape[2] < 2 and feat_cache.get(idx) is not None:
                            prev = feat_cache[idx][:, :, -1:, :, :]
                            cache_x = mx.concatenate([prev, cache_x], axis=2)
                        x = self.time_conv(x, feat_cache.get(idx))
                        if feat_idx:
                            feat_idx[0] += 1

                        # Interleave temporal dim: (B, C*2, T, H, W) -> (B, C, T*2, H, W)
                        x = x.reshape(b, 2, c, t, h, w)
                        x = mx.stack([x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]], axis=4)
                        x = x.reshape(b, c, t * 2, h, w)

            # MLX Conv2d expects NHWC (B, H, W, C), not NCHW (B, C, H, W)
            # Transpose: (B, C, T, H, W) -> (B, T, H, W, C) -> (B*T, H, W, C)
            x_nhwc = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
            x_nhwc = self.resample(x_nhwc)  # Upsample + Conv2d in NHWC
            # Output shape: (B*T, H_out, W_out, C_out)
            _, h_out, w_out, c_out = x_nhwc.shape
            # Transpose back: (B*T, H_out, W_out, C_out) -> (B, T, H_out, W_out, C_out) -> (B, C_out, T, H_out, W_out)
            x = x_nhwc.reshape(b, t, h_out, w_out, c_out).transpose(0, 4, 1, 2, 3)

            if self.mode.endswith('3d') and 'downsample' in self.mode:
                if feat_cache is not None:
                    idx = feat_idx[0] if feat_idx else 0
                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(mx.concatenate(
                        [feat_cache.get(idx, x[:, :, -1:, :, :])[:, :, -1:, :, :], x], axis=2))
                    feat_cache[idx] = cache_x
                    if feat_idx:
                        feat_idx[0] += 1

            return x
        return x


class Encoder3d(nn.Module):
    """3D VAE encoder."""

    def __init__(self, dim=128, z_dim=16, dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2, attn_scales=None,
                 temporal_downsample=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales or []
        self.temporal_downsample = temporal_downsample or [True, True, False]

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temporal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0

        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None and feat_idx is not None:
            cache_idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :]
            if cache_x.shape[2] < 2 and feat_cache.get(cache_idx) is not None:
                prev = feat_cache[cache_idx][:, :, -1:, :, :]
                cache_x = mx.concatenate([prev, cache_x], axis=2)
            x = self.conv1(x, feat_cache.get(cache_idx))
            feat_cache[cache_idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples.layers:
            if feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle.layers:
            if isinstance(layer, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head.layers:
            if isinstance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                cache_idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :]
                if cache_x.shape[2] < 2 and feat_cache.get(cache_idx) is not None:
                    prev = feat_cache[cache_idx][:, :, -1:, :, :]
                    cache_x = mx.concatenate([prev, cache_x], axis=2)
                x = layer(x, feat_cache.get(cache_idx))
                feat_cache[cache_idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    """3D VAE decoder for text-to-video inference."""

    def __init__(self, dim=128, z_dim=16, dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2, attn_scales=None,
                 temporal_upsample=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales or []
        self.temporal_upsample = temporal_upsample or [False, True, True]

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / (2 ** (len(dim_mult) - 2))

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if self.temporal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None and feat_idx is not None:
            cache_idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :]
            if cache_x.shape[2] < 2 and feat_cache.get(cache_idx) is not None:
                prev = feat_cache[cache_idx][:, :, -1:, :, :]
                cache_x = mx.concatenate([prev, cache_x], axis=2)
            x = self.conv1(x, feat_cache.get(cache_idx))
            feat_cache[cache_idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle.layers:
            if isinstance(layer, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples.layers:
            if feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head.layers:
            if isinstance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                cache_idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :]
                if cache_x.shape[2] < 2 and feat_cache.get(cache_idx) is not None:
                    prev = feat_cache[cache_idx][:, :, -1:, :, :]
                    cache_x = mx.concatenate([prev, cache_x], axis=2)
                x = layer(x, feat_cache.get(cache_idx))
                feat_cache[cache_idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class WanVAE(nn.Module):
    """VAE encoder-decoder wrapper with normalization parameters."""

    def __init__(self, z_dim=16, dim=96, dim_mult=[1, 2, 4, 4]):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.dim_mult = dim_mult

        # Normalization parameters (from PyTorch implementation)
        self.mean = mx.array([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ])
        self.std = mx.array([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ])
        self.scale = [self.mean, 1.0 / self.std]

        # Decoder
        self.decoder = Decoder3d(dim, z_dim, dim_mult)
        self.conv1 = CausalConv3d(z_dim, z_dim, 1, padding=0)

    def clear_cache(self):
        """Clear feature caches."""
        self._conv_num = 0
        self._conv_idx = [0]
        self._feat_map = {}

    def decode(self, z, scale=None):
        """Decode latents to pixel space.

        Args:
            z: Latent tensor, shape (B, z_dim, T, H, W).
            scale: Optional [mean, std] normalization parameters.

        Returns:
            Decoded pixel tensor, shape (B, 3, T, H, W), clamped to [-1, 1].
        """
        if scale is None:
            scale = self.scale

        # Denormalize: z = z / std + mean
        z = z / scale[1].reshape(1, self.z_dim, 1, 1, 1) + scale[0].reshape(1, self.z_dim, 1, 1, 1)

        # Apply 1x1 conv
        x = self.conv1(z)

        self.clear_cache()
        feat_idx = [0]

        # Frame-by-frame decode
        t = x.shape[2]
        out_cat = None
        for i in range(t):
            frame_x = x[:, :, i:i + 1, :, :]
            out = self.decoder(frame_x, feat_cache=self._feat_map, feat_idx=feat_idx)
            if out_cat is None:
                out_cat = out
            else:
                out_cat = mx.concatenate([out_cat, out], axis=2)

        return mx.clip(out_cat, -1, 1)

    def decode_batch(self, zs, scale=None):
        """Decode a batch of latent tensors."""
        if isinstance(zs, mx.array):
            b = zs.shape[0]
            outputs = []
            for i in range(b):
                out = self.decode(zs[i:i + 1], scale)
                outputs.append(out.squeeze(0))
            return outputs
        else:
            return [self.decode(u.reshape(1, *u.shape), scale).squeeze(0) for u in zs]

    def load_weights(self, weights_path):
        """Load VAE decoder weights from a .safetensors file.

        Expects PyTorch-named weights (produced by converter.py).
        Maps to MLX-compatible naming and handles Conv3d/Conv2d axis permutation.
        Only loads weights that exist in the model (skips encoder-only weights).
        """
        from mlx.utils import tree_flatten, tree_unflatten
        from safetensors.numpy import load_file

        weights = load_file(weights_path)
        mlx_items = self._mlxify_params(dict(weights))

        # Get model parameter names from tree_flatten (leaf arrays only)
        model_params = dict(tree_flatten(self))
        model_param_names = set(k for k, v in model_params.items()
                                if isinstance(v, mx.array))

        # Only keep weights that exist in the model
        filtered = []
        for k, v in mlx_items.items():
            if k not in model_param_names:
                continue
            arr = mx.array(v)
            target = model_params[k]
            # Squeeze singleton dims if shape doesn't match
            # (handles PyTorch->MLX gamma/bias shape diffs: (C,1,1) -> (C,))
            if arr.shape != target.shape and arr.ndim > target.ndim:
                arr = arr.squeeze()
            if arr.shape == target.shape:
                filtered.append((k, arr))
            else:
                print(f"  Shape mismatch: {k} weight={arr.shape} model={target.shape}, skipping")
        skipped = len(mlx_items) - len(filtered)
        if skipped:
            print(f"  Skipped {skipped} encoder-only weights")

        self.update(tree_unflatten(filtered))

    def _mlxify_params(self, pytorch_params):
        """Map PyTorch VAE parameter names to MLX names.

        Handles:
        - Sequential indices: .N. -> .layers.N.
        - CausalConv3d: convN.weight -> convN.conv.weight
        - Conv3d weight shape: (out, in, D, H, W) -> (out, D, H, W, in)
        - Conv2d weight shape: (out, in, H, W) -> (out, H, W, in)
        """
        import re
        import numpy as np
        mlx_params = {}
        for key, value in pytorch_params.items():
            ml_key = key

            # Strip model. prefix
            if ml_key.startswith("model."):
                ml_key = ml_key[6:]

            # Sequential indices: .N. -> .layers.N.
            ml_key = re.sub(r'\.(\d+)\.', r'.layers.\1.', ml_key)

            # CausalConv3d: add .conv before .weight/.bias
            # Only match decoder convs (skip encoder conv1/conv2)
            ml_key = re.sub(r'^(decoder\.conv\d+)\.(weight|bias)$', r'\1.conv.\2', ml_key)
            # WanVAE top-level conv1 (1x1 conv) — only if it's z_dim->z_dim
            if re.match(r'^conv1\.(weight|bias)$', ml_key) and value.ndim == 5:
                # The encoder's conv1 has 3 input channels, the decoder's doesn't have top-level conv1
                # Only the WanVAE wrapper has conv1 (1x1, z_dim->z_dim)
                # Skip encoder conv1 by checking if in_channels match
                pass  # handled by generic 5D detection below
            # shortcut
            ml_key = re.sub(r'^(.*\.shortcut)\.(weight|bias)$', r'\1.conv.\2', ml_key)
            # time_conv
            ml_key = re.sub(r'^(.*\.time_conv)\.(weight|bias)$', r'\1.conv.\2', ml_key)
            # residual layers (N=2 and N=6 are CausalConv3d, N=0/3/... are RMSNorm with gamma)
            if '.residual.layers.' in ml_key and ml_key.endswith(('.weight', '.bias')):
                ml_key = re.sub(r'^(.*\.residual\.layers\.\d+)\.(weight|bias)$',
                                r'\1.conv.\2', ml_key)

            # Generic: any 5D .weight at a Sequential layer index is a CausalConv3d
            # (e.g. head.layers.2.weight -> head.layers.2.conv.weight)
            if ml_key.endswith('.weight') and value.ndim == 5:
                match = re.match(r'^(.*\.layers\.\d+)\.(weight|bias)$', ml_key)
                if match and '.conv.' not in ml_key:
                    ml_key = f'{match.group(1)}.conv.{match.group(2)}'
            # Corresponding bias for 5D Conv3d at Sequential layer index
            if ml_key.endswith('.bias') and '.conv.' not in ml_key:
                match = re.match(r'^(.*\.layers\.(\d+))\.bias$', ml_key)
                if match and int(match.group(2)) % 2 == 0:
                    ml_key = f'{match.group(1)}.conv.bias'

            # Handle Conv3d weight shape permutation
            if ml_key.endswith('.weight') and value.ndim == 5:
                # PyTorch Conv3d: (out, in, D, H, W) -> MLX: (out, D, H, W, in)
                value = np.ascontiguousarray(value.transpose(0, 2, 3, 4, 1))

            # Handle Conv2d weight shape permutation (not gamma)
            if ml_key.endswith('.weight') and value.ndim == 4:
                # PyTorch Conv2d: (out, in, H, W) -> MLX: (out, H, W, in)
                value = np.ascontiguousarray(value.transpose(0, 2, 3, 1))

            mlx_params[ml_key] = value
        return mlx_params
