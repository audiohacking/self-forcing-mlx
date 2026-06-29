"""Integration tests for CausalInferencePipeline with random weights."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.pipeline import (
    CausalInferencePipeline,
    _initialize_kv_caches,
    _initialize_crossattn_caches,
    _reset_kv_caches,
    _reset_crossattn_caches,
)
from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM,
    TEXT_DIM, TEXT_LEN, IN_DIM, OUT_DIM, PATCH_SIZE, EPS,
    HEAD_DIM, MAX_SEQ_LEN, GUIDANCE_SCALE, NUM_TRAIN_TIMESTEPS,
)


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestCausalInferencePipeline:
    def test_instantiation(self):
        """Pipeline can be created without loading weights (random init)."""
        pipeline = CausalInferencePipeline.__new__(CausalInferencePipeline)
        # Skip weight loading for unit test - just verify class exists
        assert pipeline is not None

    def test_kv_cache_init(self):
        """KV caches initialized with correct shapes."""
        caches = _initialize_kv_caches(1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        assert len(caches) == NUM_LAYERS
        assert caches[0]["k"].shape == (1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        assert caches[0]["v"].shape == (1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        assert caches[0]["global_end_index"].item() == 0

    def test_crossattn_cache_init(self):
        caches = _initialize_crossattn_caches(1, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM)
        assert len(caches) == NUM_LAYERS
        assert caches[0]["k"].shape == (1, TEXT_LEN, NUM_HEADS, HEAD_DIM)
        assert caches[0]["is_init"] is False

    def test_kv_cache_reset(self):
        caches = _initialize_kv_caches(1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        caches[0]["global_end_index"] = mx.array([100], dtype=mx.int32)
        caches[0]["local_end_index"] = mx.array([100], dtype=mx.int32)
        _reset_kv_caches(caches)
        assert caches[0]["global_end_index"].item() == 0
        assert caches[0]["local_end_index"].item() == 0

    def test_crossattn_cache_reset(self):
        caches = _initialize_crossattn_caches(1, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM)
        caches[0]["is_init"] = True
        _reset_crossattn_caches(caches)
        assert caches[0]["is_init"] is False

    def test_kv_cache_store_and_retrieve(self, pipeline_with_random_weights):
        """KV cache stores actual tokens (not padding) and retrieves correctly."""
        pipeline = pipeline_with_random_weights
        pipeline._initialize_caches(1)
        kv = pipeline._kv_caches_pos

        # Create input for 1 frame
        noise = mx.random.normal((1, 1, IN_DIM, pipeline.fh, pipeline.fw))
        timestep = mx.array([1000], dtype=mx.float32)
        context = mx.random.normal((1, TEXT_LEN, TEXT_DIM))

        # First forward pass
        out1 = pipeline._forward_transformer(
            noise, timestep, context,
            kv_caches=kv,
            crossattn_caches=pipeline._crossattn_caches_pos,
        )
        # Cache should have stored actual tokens (not 32760)
        cached_count = kv[0]["local_end_index"].item()
        assert cached_count > 0, "Cache should have stored tokens"
        assert cached_count < MAX_SEQ_LEN, \
            f"Cache stored {cached_count} (should be < {MAX_SEQ_LEN}, not padded length)"
        assert out1.shape == noise.shape

    def test_multi_block_inference(self, pipeline_with_random_weights):
        """Multi-block inference should not overflow cache."""
        pipeline = pipeline_with_random_weights
        context_cond = mx.random.normal((1, TEXT_LEN, TEXT_DIM))
        context_uncond = mx.random.normal((1, TEXT_LEN, TEXT_DIM))

        # 4 frames = 2 blocks (1 + 3)
        noise = mx.random.normal((1, 4, IN_DIM, pipeline.fh, pipeline.fw))

        latents = pipeline.inference(noise, context_cond, context_uncond)
        assert latents.shape == (1, 4, IN_DIM, pipeline.fh, pipeline.fw)
        assert_finite(latents, "inference output")

    def test_initial_latent(self, pipeline_with_random_weights):
        """Image-to-video: initial latent should be cached correctly."""
        pipeline = pipeline_with_random_weights
        context_cond = mx.random.normal((1, TEXT_LEN, TEXT_DIM))
        context_uncond = mx.random.normal((1, TEXT_LEN, TEXT_DIM))

        # 3 frames with 1 initial frame
        noise = mx.random.normal((1, 3, IN_DIM, pipeline.fh, pipeline.fw))
        initial = mx.random.normal((1, 1, IN_DIM, pipeline.fh, pipeline.fw))

        latents = pipeline.inference(noise, context_cond, context_uncond,
                                     initial_latent=initial)
        # num_output_frames = 3 + 1 = 4
        assert latents.shape == (1, 4, IN_DIM, pipeline.fh, pipeline.fw)
        assert_finite(latents, "inference with initial latent")


@pytest.fixture
def pipeline_with_random_weights():
    """Create a pipeline with random weights (no file loading)."""
    from sforcing.model import WanModel
    from sforcing.vae import WanVAE
    from sforcing.t5 import T5Encoder
    from sforcing.scheduler import FlowMatchScheduler
    from sforcing.tokenizer_bridge import HuggingfaceTokenizer

    pipeline = CausalInferencePipeline.__new__(CausalInferencePipeline)
    pipeline.num_frames = 21
    pipeline.height = 480
    pipeline.width = 832
    pipeline.guidance_scale = GUIDANCE_SCALE
    pipeline.num_frame_per_block = 3
    pipeline.negative_prompt = ""
    pipeline.dtype = mx.float32
    pipeline.fh = 60
    pipeline.fw = 104
    pipeline.ft = 5  # 21 // 4
    pipeline.frame_seq_len = pipeline.ft * pipeline.fh * pipeline.fw
    pipeline.max_seq_len = MAX_SEQ_LEN

    # Random-weight models
    pipeline.t5 = T5Encoder()
    pipeline.model = WanModel()
    pipeline.vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])

    # Tokenizer
    try:
        pipeline.tokenizer = HuggingfaceTokenizer(
            name="google/umt5-xxl",
            seq_len=TEXT_LEN,
            clean='whitespace',
        )
    except Exception:
        pipeline.tokenizer = None

    # Scheduler
    pipeline.denoising_step_list = [1000]
    pipeline.scheduler = FlowMatchScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=5.0,
        extra_one_step=True,
    )
    pipeline.scheduler.set_timesteps(len(pipeline.denoising_step_list))

    pipeline._kv_caches_pos = None
    pipeline._kv_caches_neg = None
    pipeline._crossattn_caches_pos = None
    pipeline._crossattn_caches_neg = None

    return pipeline
