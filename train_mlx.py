#!/usr/bin/env python3
"""MLX training script for Self-Forcing video diffusion.

Trains the generator using one of the supported training strategies:
  - diffusion    : Standard flow-matching denoising loss
  - dmd          : Distribution Matching Distillation
  - causvid      : CausVid DMD (simplified)
  - gan          : GAN with discriminator
  - sid          : Score identity Distillation
  - ode          : ODE regression from precomputed trajectories

Usage:
    python train_mlx.py --config configs/my_config.yaml
    python train_mlx.py --mode diffusion --data /path/to/lmdb --checkpoint /path/to/weights
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM, TEXT_DIM, TEXT_LEN,
    IN_DIM, OUT_DIM, PATCH_SIZE, EPS, NUM_TRAIN_TIMESTEPS,
    T5_VOCAB_SIZE, T5_DIM, T5_DIM_ATTN, T5_DIM_FFN,
    T5_NUM_HEADS, T5_ENCODER_LAYERS, T5_NUM_BUCKETS,
    T5_SHARED_POS, T5_DROPOUT,
)
from sforcing.model import WanModel
from sforcing.t5 import T5Encoder
from sforcing.scheduler import FlowMatchScheduler
from sforcing.data import ShardingLMDBDataset, ODERegressionLMDBDataset, cycle, set_seed
from sforcing.trainers import (
    CausalDiffusion, CausVid, DMD, GAN, SiD, ODERegression,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configs. Install with: pip install pyyaml")
    elif config_path.endswith(".json"):
        with open(config_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Load weights from a .safetensors file."""
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            params[key] = np.array(f.get_tensor(key))
    return params


# ---------------------------------------------------------------------------
# Training step functions
# ---------------------------------------------------------------------------

def train_step_diffusion(
    model: CausalDiffusion,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    key: mx.array,
) -> mx.array:
    """Single training step for diffusion training."""

    def loss_fn(params, clean_latent, conditional_dict, key):
        model.generator.update(params)
        return model.loss_fn(clean_latent, conditional_dict, key=key)

    clean_latent = batch.get("ode_latent", batch.get("latent"))
    if clean_latent.ndim == 6:  # (B, steps, F, C, H, W) -> take last (cleanest)
        clean_latent = clean_latent[:, -1]

    # Build conditional dict from text
    prompts = batch.get("prompts", [""] * clean_latent.shape[0])
    conditional_dict = {"context": mx.ones((len(prompts), TEXT_LEN, TEXT_DIM))}

    loss_and_grad_fn = nn.value_and_grad(model.generator, loss_fn)
    loss, grads = loss_and_grad_fn(
        model.generator.trainable_parameters(),
        clean_latent, conditional_dict, key,
    )
    optimizer.update(model.generator, grads)
    return loss


def train_step_dmd(
    model: DMD,
    gen_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    key: mx.array,
) -> Dict[str, mx.array]:
    """Single training step for DMD training."""
    clean_latent = batch.get("ode_latent", batch.get("latent"))
    if clean_latent.ndim == 6:
        clean_latent = clean_latent[:, -1]

    prompts = batch.get("prompts", [""] * clean_latent.shape[0])
    conditional_dict = {"context": mx.ones((len(prompts), TEXT_LEN, TEXT_DIM))}
    unconditional_dict = {"context": mx.zeros((len(prompts), TEXT_LEN, TEXT_DIM))}

    # Generator step
    def gen_loss_fn(generator_params):
        model.generator.update(generator_params)
        loss, _ = model.generator_loss(clean_latent, conditional_dict, unconditional_dict, key=key)
        return loss

    gen_loss, gen_grads = nn.value_and_grad(model.generator, gen_loss_fn)(
        model.generator.trainable_parameters(),
    )
    gen_optimizer.update(model.generator, gen_grads)

    # Critic step
    def critic_loss_fn(fake_score_params):
        model.fake_score.update(fake_score_params)
        loss, _ = model.critic_loss(clean_latent, conditional_dict, unconditional_dict, key=key)
        return loss

    critic_loss, critic_grads = nn.value_and_grad(model.fake_score, critic_loss_fn)(
        model.fake_score.trainable_parameters(),
    )
    critic_optimizer.update(model.fake_score, critic_grads)

    return {"gen_loss": gen_loss, "critic_loss": critic_loss}


def train_step_gan(
    model: GAN,
    gen_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    key: mx.array,
) -> Dict[str, mx.array]:
    """Single training step for GAN training."""
    clean_latent = batch.get("ode_latent", batch.get("latent"))
    if clean_latent.ndim == 6:
        clean_latent = clean_latent[:, -1]

    prompts = batch.get("prompts", [""] * clean_latent.shape[0])
    conditional_dict = {"context": mx.ones((len(prompts), TEXT_LEN, TEXT_DIM))}
    unconditional_dict = {"context": mx.zeros((len(prompts), TEXT_LEN, TEXT_DIM))}

    # Generator step
    def gen_loss_fn(generator_params):
        model.generator.update(generator_params)
        return model.generator_loss(clean_latent, conditional_dict, unconditional_dict, key=key)

    gen_loss, gen_grads = nn.value_and_grad(model.generator, gen_loss_fn)(
        model.generator.trainable_parameters(),
    )
    gen_optimizer.update(model.generator, gen_grads)

    # Critic step
    def critic_loss_fn(fake_score_params):
        model.fake_score.update(fake_score_params)
        losses, _ = model.critic_loss(
            clean_latent, clean_latent, conditional_dict, unconditional_dict, key=key,
        )
        return losses[0] + losses[1] + losses[2]  # gan_d + r1 + r2

    critic_loss, critic_grads = nn.value_and_grad(model.fake_score, critic_loss_fn)(
        model.fake_score.trainable_parameters(),
    )
    critic_optimizer.update(model.fake_score, critic_grads)

    return {"gen_loss": gen_loss, "critic_loss": critic_loss}


def train_step_sid(
    model: SiD,
    gen_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    key: mx.array,
) -> Dict[str, mx.array]:
    """Single training step for SiD training."""
    clean_latent = batch.get("ode_latent", batch.get("latent"))
    if clean_latent.ndim == 6:
        clean_latent = clean_latent[:, -1]

    prompts = batch.get("prompts", [""] * clean_latent.shape[0])
    conditional_dict = {"context": mx.ones((len(prompts), TEXT_LEN, TEXT_DIM))}
    unconditional_dict = {"context": mx.zeros((len(prompts), TEXT_LEN, TEXT_DIM))}

    # Generator step
    def gen_loss_fn(generator_params):
        model.generator.update(generator_params)
        loss, _ = model.generator_loss(clean_latent, conditional_dict, unconditional_dict, key=key)
        return loss

    gen_loss, gen_grads = nn.value_and_grad(model.generator, gen_loss_fn)(
        model.generator.trainable_parameters(),
    )
    gen_optimizer.update(model.generator, gen_grads)

    # Critic step
    def critic_loss_fn(fake_score_params):
        model.fake_score.update(fake_score_params)
        loss, _ = model.critic_loss(clean_latent, conditional_dict, unconditional_dict, key=key)
        return loss

    critic_loss, critic_grads = nn.value_and_grad(model.fake_score, critic_loss_fn)(
        model.fake_score.trainable_parameters(),
    )
    critic_optimizer.update(model.fake_score, critic_grads)

    return {"gen_loss": gen_loss, "critic_loss": critic_loss}


def train_step_ode(
    model: ODERegression,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    key: mx.array,
) -> mx.array:
    """Single training step for ODE regression training."""

    def loss_fn(params, ode_latent, conditional_dict, key):
        model.generator.update(params)
        return model.loss_fn(ode_latent, conditional_dict, key=key)

    ode_latent = batch["ode_latent"]
    prompts = batch.get("prompts", [""] * ode_latent.shape[0])
    conditional_dict = {"context": mx.ones((len(prompts), TEXT_LEN, TEXT_DIM))}

    loss, grads = nn.value_and_grad(model.generator, loss_fn)(
        model.generator.trainable_parameters(),
        ode_latent, conditional_dict, key,
    )
    optimizer.update(model.generator, grads)
    return loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    """Run the training loop."""
    set_seed(args.seed)

    # --- Load config ---
    config = {}
    if args.config:
        config = load_config(args.config)

    # Command-line overrides
    mode = args.mode or config.get("trainer", "diffusion")
    data_path = args.data or config.get("data_path", "")
    checkpoint_dir = args.checkpoint or config.get("checkpoint_dir", "./mlx_weights")
    lr = args.lr or config.get("lr", 2e-6)
    lr_critic = args.lr_critic or config.get("lr_critic", 4e-7)
    num_steps = args.steps or config.get("num_steps", 1000)
    log_interval = args.log_interval or config.get("log_iters", 50)
    save_interval = args.save_interval or config.get("log_iters", 500)
    batch_size = args.batch_size or config.get("batch_size", 1)
    seed = args.seed or config.get("seed", 42)

    print(f"=== Self-Forcing MLX Training ===")
    print(f"Mode: {mode}")
    print(f"Data: {data_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Learning rate: gen={lr}, critic={lr_critic}")
    print(f"Steps: {num_steps}")

    # --- Build models ---
    print("\nBuilding models...")

    # Scheduler
    scheduler = FlowMatchScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=config.get("timestep_shift", 5.0),
        extra_one_step=True,
    )

    # Generator
    generator = WanModel(
        dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM, text_len=TEXT_LEN,
        in_dim=IN_DIM, out_dim=OUT_DIM,
        patch_size=PATCH_SIZE, eps=EPS,
    )

    # Load generator weights
    gen_path = os.path.join(checkpoint_dir, "transformer.safetensors")
    if os.path.exists(gen_path):
        print(f"Loading generator from {gen_path}")
        params = load_safetensors(gen_path)
        generator.load_weights(gen_path)
    else:
        print(f"Warning: generator weights not found at {gen_path}")

    # For DMD/GAN/SiD: create real_score and fake_score
    real_score = None
    fake_score = None

    if mode in ("dmd", "causvid", "gan", "sid"):
        real_score = WanModel(
            dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, freq_dim=FREQ_DIM,
            text_dim=TEXT_DIM, text_len=TEXT_LEN,
            in_dim=IN_DIM, out_dim=OUT_DIM,
            patch_size=PATCH_SIZE, eps=EPS,
        )
        fake_score = WanModel(
            dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, freq_dim=FREQ_DIM,
            text_dim=TEXT_DIM, text_len=TEXT_LEN,
            in_dim=IN_DIM, out_dim=OUT_DIM,
            patch_size=PATCH_SIZE, eps=EPS,
        )

        # Load real_score from same checkpoint (frozen)
        if os.path.exists(gen_path):
            real_score.load_weights(gen_path)
        # fake_score starts from same weights (trainable)

        # Freeze real_score
        real_score.freeze()

    # --- Build trainer ---
    num_frame_per_block = config.get("num_frame_per_block", 3)
    num_train_timestep = config.get("num_train_timestep", 1000)
    min_step = config.get("min_step", 0.02)
    max_step = config.get("max_step", 0.98)
    guidance_scale = config.get("guidance_scale", 5.0)

    if mode == "diffusion":
        trainer = CausalDiffusion(
            generator=generator, scheduler=scheduler,
            num_frame_per_block=num_frame_per_block,
            num_train_timestep=num_train_timestep,
            min_step=min_step, max_step=max_step,
        )
    elif mode == "dmd":
        trainer = DMD(
            generator=generator, real_score=real_score, fake_score=fake_score,
            scheduler=scheduler, num_frame_per_block=num_frame_per_block,
            num_train_timestep=num_train_timestep,
            min_step=min_step, max_step=max_step,
            real_guidance_scale=guidance_scale,
        )
    elif mode == "causvid":
        trainer = CausVid(
            generator=generator, real_score=real_score, fake_score=fake_score,
            scheduler=scheduler, num_frame_per_block=num_frame_per_block,
            num_train_timestep=num_train_timestep,
            min_step=min_step, max_step=max_step,
            real_guidance_scale=guidance_scale,
        )
    elif mode == "gan":
        trainer = GAN(
            generator=generator, real_score=real_score, fake_score=fake_score,
            scheduler=scheduler, num_frame_per_block=num_frame_per_block,
            num_train_timestep=num_train_timestep,
            min_step=min_step, max_step=max_step,
            real_guidance_scale=guidance_scale,
        )
    elif mode == "sid":
        trainer = SiD(
            generator=generator, real_score=real_score, fake_score=fake_score,
            scheduler=scheduler, num_frame_per_block=num_frame_per_block,
            num_train_timestep=num_train_timestep,
            min_step=min_step, max_step=max_step,
            real_guidance_scale=guidance_scale,
        )
    elif mode == "ode":
        trainer = ODERegression(
            generator=generator, num_frame_per_block=num_frame_per_block,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Trainer: {type(trainer).__name__}")

    # --- Setup optimizers ---
    gen_optimizer = optim.AdamW(learning_rate=lr, betas=(0.0, 0.999))
    critic_optimizer = optim.AdamW(learning_rate=lr_critic, betas=(0.0, 0.999)) \
        if mode in ("dmd", "causvid", "gan", "sid") else None

    # --- Setup data ---
    dataset = None
    if data_path and os.path.exists(data_path):
        if mode == "ode":
            dataset = ODERegressionLMDBDataset(data_path)
        else:
            dataset = ShardingLMDBDataset(data_path)
        print(f"Dataset size: {len(dataset)}")

    # --- Training loop ---
    print(f"\nStarting training for {num_steps} steps...\n")
    key = mx.random.key(seed)
    step = 0
    start_time = time.time()

    while step < num_steps:
        key, step_key = mx.random.split(key)

        # Get batch
        if dataset is not None:
            idx = step % len(dataset)
            batch = dataset[idx]
        else:
            # Dummy batch for testing
            batch = {
                "ode_latent": mx.random.normal((1, 4, 21, IN_DIM, 60, 104)),
                "prompts": ["test prompt"],
            }

        # Training step
        if mode == "diffusion":
            loss = train_step_diffusion(trainer, gen_optimizer, batch, step_key)
        elif mode == "dmd":
            losses = train_step_dmd(trainer, gen_optimizer, critic_optimizer, batch, step_key)
        elif mode == "causvid":
            losses = train_step_dmd(trainer, gen_optimizer, critic_optimizer, batch, step_key)
        elif mode == "gan":
            losses = train_step_gan(trainer, gen_optimizer, critic_optimizer, batch, step_key)
        elif mode == "sid":
            losses = train_step_sid(trainer, gen_optimizer, critic_optimizer, batch, step_key)
        elif mode == "ode":
            loss = train_step_ode(trainer, gen_optimizer, batch, step_key)

        step += 1

        # Logging
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0

            if mode in ("dmd", "causvid", "gan", "sid"):
                gen_loss_val = losses["gen_loss"].item()
                critic_loss_val = losses["critic_loss"].item()
                print(f"Step {step:6d}/{num_steps} | "
                      f"gen_loss: {gen_loss_val:.4f} | "
                      f"critic_loss: {critic_loss_val:.4f} | "
                      f"{steps_per_sec:.2f} steps/s")
            else:
                loss_val = loss.item()
                print(f"Step {step:6d}/{num_steps} | "
                      f"loss: {loss_val:.6f} | "
                      f"{steps_per_sec:.2f} steps/s")

        # Save checkpoint
        if step % save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"generator_step_{step:06d}.safetensors")
            print(f"\nSaving checkpoint to {save_path}")
            # TODO: implement safetensors save of MLX model weights

    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLX Self-Forcing Training")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["diffusion", "dmd", "causvid", "gan", "sid", "ode"],
                        help="Training mode (overrides config)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to LMDB dataset (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory with safetensors files")
    parser.add_argument("--lr", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-critic", type=float, default=None, help="Critic learning rate")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=None, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=None, help="Save every N steps")
    parser.add_argument("--output-dir", type=str, default="./mlx_checkpoints",
                        help="Output directory for checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
