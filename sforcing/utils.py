"""Math utility functions ported from PyTorch.

Ported from wan/modules/model.py and wan/modules/causal_model.py.
Uses mlx-arsenal for RoPE where applicable.
"""
import mlx.core as mx


def sinusoidal_embedding_1d(dim, position):
    """Sinusoidal 1D embedding for time/position.

    Args:
        dim: Embedding dimension (must be even).
        position: Position tensor, shape (...,).

    Returns:
        Embedding tensor, shape (..., dim).
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)
    freqs = mx.power(10000.0, -mx.arange(half, dtype=mx.float32).astype(position.dtype) / half)
    sinusoid = mx.outer(position, freqs)
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    """Precompute RoPE frequencies as complex numbers.

    Args:
        max_seq_len: Maximum sequence length.
        dim: Dimension (must be even).
        theta: Theta parameter for frequency computation.

    Returns:
        Frequencies tensor, shape (max_seq_len, dim//2) as complex64.
    """
    assert dim % 2 == 0
    freqs = mx.outer(
        mx.arange(max_seq_len, dtype=mx.float32),
        1.0 / mx.power(theta,
                        mx.arange(0, dim, 2, dtype=mx.float32).astype(mx.float32) / dim),
    )
    # Return as complex: cos + i*sin
    return mx.exp(1j * freqs)


def rope_apply(x, grid_sizes, freqs):
    """Apply 3D RoPE to query/key tensors.

    Each token has a (t, h, w) position. The head dim is split into
    three parts: temporal, height, width. Each part gets 1D RoPE.

    Args:
        x: Query/key tensor, shape (B, L, num_heads, head_dim).
        grid_sizes: Grid dimensions tensor, shape (B, 3) with (F, H, W).
        freqs: Precomputed frequencies, shape (max_seq, D), complex64
               where D = sum(dims_per_axis) and D = head_dim // 2.

    Returns:
        RoPE-applied tensor, same shape as x.
    """
    n, c = x.shape[2], x.shape[3] // 2  # n=num_heads, c=head_dim/2

    # Split freqs into temporal, height, width components
    # dims_per_axis = [d - 4*(d//6), 2*(d//6), 2*(d//6)] where d = head_dim
    d = x.shape[3]  # head_dim
    t_dim = d - 4 * (d // 6)  # temporal portion
    s_dim = 2 * (d // 6)      # spatial (h/w) portion
    # In frequency space, each real dim maps to half complex dim
    # MLX split with list uses indices (numpy-style), not sizes (PyTorch-style).
    # Compute cumulative indices for correct splitting.
    split_sizes = [t_dim // 2, s_dim // 2, s_dim // 2]
    split_indices = [sum(split_sizes[:i]) for i in range(1, len(split_sizes))]
    ft, fh, fw = mx.split(freqs, split_indices, axis=1)

    results = []
    for i in range(x.shape[0]):
        f, h, w = int(grid_sizes[i, 0]), int(grid_sizes[i, 1]), int(grid_sizes[i, 2])
        seq_len = f * h * w

        # Convert to complex: (seq_len, n, head_dim/2) as complex
        x_i = x[i, :seq_len].astype(mx.float32)
        x_i = x_i.reshape(seq_len, n, -1, 2)
        x_i = x_i[..., 0] + 1j * x_i[..., 1]  # (seq_len, n, c) complex

        # Build per-position frequency multipliers by broadcasting
        # ft[:f]: (f, t_dim/2) -> expand to (f, h, w, t_dim/2)
        # fh[:h]: (h, s_dim/2) -> expand to (f, h, w, s_dim/2)
        # fw[:w]: (w, s_dim/2) -> expand to (f, h, w, s_dim/2)
        # Concatenate along last dim: (f, h, w, c) where c = head_dim/2
        freqs_i = mx.concatenate([
            mx.broadcast_to(ft[:f].reshape(f, 1, 1, -1), (f, h, w, ft.shape[-1])),
            mx.broadcast_to(fh[:h].reshape(1, h, 1, -1), (f, h, w, fh.shape[-1])),
            mx.broadcast_to(fw[:w].reshape(1, 1, w, -1), (f, h, w, fw.shape[-1])),
        ], axis=-1).reshape(seq_len, 1, -1)  # (seq_len, 1, c)

        # Apply rotation via complex multiplication
        x_i = x_i * freqs_i  # (seq_len, n, c) complex

        # Convert back to real: (seq_len, n, c) complex -> (seq_len, n, c, 2) -> (seq_len, n, c*2)
        x_i = mx.stack([x_i.real, x_i.imag], axis=-1)
        x_i = x_i.reshape(seq_len, n, -1)  # (seq_len, n, head_dim)

        # Append remainder (padded tokens) if any
        if seq_len < x.shape[1]:
            x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)
        else:
            x_i = x_i

        results.append(x_i)

    return mx.stack(results).astype(x.dtype)


def rope_apply_causal(x, grid_sizes, freqs, start_frame=0):
    """Causal variant of RoPE with frame offset.

    Used during autoregressive inference where each new frame starts
    at a different temporal position in the frequency embedding.

    Args:
        x: Query/key tensor, shape (B, L, num_heads, head_dim).
        grid_sizes: Grid dimensions tensor, shape (B, 3) with (F, H, W).
        freqs: Precomputed frequencies, shape (max_seq, D), complex64.
        start_frame: Starting frame offset for causal attention.

    Returns:
        RoPE-applied tensor, same shape as x.
    """
    n, c = x.shape[2], x.shape[3] // 2
    d = x.shape[3]
    t_dim = d - 4 * (d // 6)
    s_dim = 2 * (d // 6)

    split_sizes = [t_dim // 2, s_dim // 2, s_dim // 2]
    split_indices = [sum(split_sizes[:i]) for i in range(1, len(split_sizes))]
    ft, fh, fw = mx.split(freqs, split_indices, axis=1)

    results = []
    for i in range(x.shape[0]):
        f, h, w = int(grid_sizes[i, 0]), int(grid_sizes[i, 1]), int(grid_sizes[i, 2])
        seq_len = f * h * w

        x_i = x[i, :seq_len].astype(mx.float32)
        x_i = x_i.reshape(seq_len, n, -1, 2)
        x_i = x_i[..., 0] + 1j * x_i[..., 1]

        # Use start_frame offset for temporal component
        freqs_i = mx.concatenate([
            mx.broadcast_to(ft[start_frame:start_frame + f].reshape(f, 1, 1, -1), (f, h, w, ft.shape[-1])),
            mx.broadcast_to(fh[:h].reshape(1, h, 1, -1), (f, h, w, fh.shape[-1])),
            mx.broadcast_to(fw[:w].reshape(1, 1, w, -1), (f, h, w, fw.shape[-1])),
        ], axis=-1).reshape(seq_len, 1, -1)

        x_i = x_i * freqs_i
        x_i = mx.stack([x_i.real, x_i.imag], axis=-1)
        x_i = x_i.reshape(seq_len, n, -1)

        if seq_len < x.shape[1]:
            x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)

        results.append(x_i)

    return mx.stack(results).astype(x.dtype)


def unpatchify(x, grid_sizes, patch_size, out_dim):
    """Reconstruct video tensors from patch embeddings.

    Args:
        x: Patchified features, shape (B, L, out_dim * prod(patch_size)).
        grid_sizes: Grid dimensions, shape (B, 3) with (F_patches, H_patches, W_patches).
        patch_size: Patch dimensions (t, h, w).
        out_dim: Output channels per patch element.

    Returns:
        Reconstructed tensors, list of shape (B, out_dim, F, H/8, W/8).
    """
    out = []
    for u, v in zip(x, grid_sizes):
        f_p, h_p, w_p = [int(s) for s in v]
        prod_vp = f_p * h_p * w_p

        u = u[:prod_vp]
        u = u.reshape(f_p, h_p, w_p, *patch_size, out_dim)

        # einsum: fhwpqrc -> cfphqwr
        # u[f, h, w, p_t, p_h, p_w, c] -> out[c, f*p_t, h*p_h, w*p_w]
        perm = [3, 0, 4, 1, 5, 2, 6]  # (p_t, f, p_h, h, p_w, w, c)
        u = u.transpose(perm)
        u = u.reshape(out_dim, f_p * patch_size[0], h_p * patch_size[1], w_p * patch_size[2])
        out.append(u)

    return out


def gelu_tanh(x):
    """GELU approximation using tanh (faster on Metal).

    From PyTorch's nn.GELU(approximate='tanh').

    Args:
        x: Input tensor.

    Returns:
        GELU-applied tensor.
    """
    return 0.5 * x * (1.0 + mx.tanh(
        mx.sqrt(2.0 / mx.pi) * (x + 0.044715 * mx.power(x, 3.0))
    ))


def silu(x):
    """SiLU/Swish activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + mx.exp(-x)))


def fp16_clamp(x, max_val=65504.0):
    """Clamp inf values in float16/bfloat16.

    Args:
        x: Input tensor.
        max_val: Clamp maximum value.

    Returns:
        Clamped tensor.
    """
    if x.dtype in [mx.float16, mx.bfloat16]:
        mask = mx.isinf(x)
        x = mx.where(mask, mx.array(max_val - 1000, dtype=x.dtype), x)
    return x
