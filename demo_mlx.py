#!/usr/bin/env python3
"""MLX demo for Self-Forcing video generation with real-time frame streaming.

Adapted from demo.py (CUDA) for Apple Silicon / MLX.
Uses Flask + SocketIO for WebSocket-based frame streaming to the browser.

Usage:
    python demo_mlx.py --port 5001
    python demo_mlx.py --weights ./mlx_weights --port 5001
"""

import argparse
import base64
import os
import random
import re
import subprocess
import time
from io import BytesIO
from threading import Event, Thread

import mlx.core as mx
import numpy as np
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sforcing.config import (
    DIM, IN_DIM, TEXT_DIM, TEXT_LEN,
    NUM_HEADS, NUM_LAYERS, HEAD_DIM, MAX_SEQ_LEN,
    NUM_TRAIN_TIMESTEPS, GUIDANCE_SCALE,
)
from sforcing.pipeline import (
    CausalInferencePipeline,
    _initialize_kv_caches,
    _initialize_crossattn_caches,
    _reset_kv_caches,
    _reset_crossattn_caches,
)
from sforcing.data import set_seed

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5001)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--weights", type=str, default="./mlx_weights",
                    help="Directory with transformer.safetensors, t5_encoder.safetensors, vae_decoder.safetensors")
parser.add_argument("--num-frames", type=int, default=21, help="Number of output frames")
parser.add_argument("--height", type=int, default=480, help="Output height")
parser.add_argument("--width", type=int, default=832, help="Output width")
parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE, help="CFG guidance scale")
parser.add_argument("--num-frame-per-block", type=int, default=3, help="Frames per denoising block")
parser.add_argument("--negative-prompt", type=str,
                    default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
                    help="Negative prompt for CFG")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

print(f"MLX GPU available: {mx.metal is not None}")
if hasattr(mx, 'metal') and mx.metal is not None:
    print(f"Metal device: {mx.metal.device_info()}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

transformer_path = os.path.join(args.weights, "transformer.safetensors")
t5_path = os.path.join(args.weights, "t5_encoder.safetensors")
vae_path = os.path.join(args.weights, "vae_decoder.safetensors")

# Check if weights exist
if not all(os.path.exists(p) for p in [transformer_path, t5_path, vae_path]):
    print(f"Weights not found in {args.weights}")
    print(f"Run: python scripts/download_mlx_models.py --output {args.weights}")
    print("Or place these files:")
    for p in [transformer_path, t5_path, vae_path]:
        print(f"  {p}")
    sys.exit(1)

print("\nLoading MLX models...")
print(f"  Transformer: {transformer_path}")
print(f"  T5 encoder:  {t5_path}")
print(f"  VAE decoder: {vae_path}")

# Create the pipeline (loads all models)
pipeline = CausalInferencePipeline(
    transformer_path=transformer_path,
    t5_path=t5_path,
    vae_path=vae_path,
    num_frames=args.num_frames,
    height=args.height,
    width=args.width,
    guidance_scale=args.guidance_scale,
    num_frame_per_block=args.num_frame_per_block,
    negative_prompt=args.negative_prompt,
)

print("\n✅ Models loaded successfully!")

# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "mlx_demo_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

generation_active = False
stop_event = Event()
anim_name = ""
frame_rate = 6

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def tensor_to_base64_frame(frame_array: np.ndarray) -> str:
    """Convert a single frame (CHW, [-1,1]) to base64 JPEG string."""
    # Normalize [-1, 1] -> [0, 255]
    frame = np.clip(frame_array * 127.5 + 127.5, 0, 255).astype(np.uint8)
    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    elif frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    image = Image.fromarray(frame, "RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def save_frame(frame_array: np.ndarray, frame_index: int):
    """Save a single frame to disk."""
    frame = np.clip(frame_array * 127.5 + 127.5, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    image = Image.fromarray(frame, "RGB")
    os.makedirs(f"./images/{anim_name}", exist_ok=True)
    image.save(f"./images/{anim_name}/{anim_name}_{frame_index:03d}.jpg", quality=95)


def generate_mp4_from_images(fps: int = 6):
    """Compile saved frames into an MP4 video using ffmpeg."""
    output_path = f"./videos/{anim_name}.mp4"
    os.makedirs("./videos", exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"./images/{anim_name}/{anim_name}_%03d.jpg",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e}")


def calculate_sha256(data: str) -> str:
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()


def emit_progress(message: str, progress: int, job_id: str):
    try:
        socketio.emit("progress", {
            "message": message,
            "progress": progress,
            "job_id": job_id,
        })
    except Exception as e:
        print(f"Progress emit error: {e}")


# ---------------------------------------------------------------------------
# Generation function (runs in background thread)
# ---------------------------------------------------------------------------


def generate_video_stream(prompt: str, seed: int):
    """Generate video and stream frames to frontend in real-time."""
    global generation_active, stop_event, anim_name, frame_rate

    try:
        generation_active = True
        stop_event.clear()

        job_id = f"{seed}_{int(time.time())}"
        emit_progress("Starting generation...", 0, job_id)

        # Seed
        mx.random.seed(seed)

        # --- Encode text ---
        emit_progress("Encoding text prompt...", 5, job_id)
        print(f"Encoding prompt: '{prompt[:60]}...'")

        context_cond = pipeline._encode_text(prompt)
        context_uncond = pipeline._encode_text(pipeline.negative_prompt)
        print(f"  Context shapes - cond: {context_cond.shape}, uncond: {context_uncond.shape}")

        # --- Create noise ---
        emit_progress("Initializing generation...", 10, job_id)
        noise = mx.random.normal(
            (1, pipeline.num_frames, IN_DIM, pipeline.fh, pipeline.fw),
            dtype=pipeline.dtype,
        )

        # --- Initialize KV caches ---
        kv_caches_pos = _initialize_kv_caches(
            1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, pipeline.dtype)
        kv_caches_neg = _initialize_kv_caches(
            1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, pipeline.dtype)
        crossattn_caches_pos = _initialize_crossattn_caches(
            1, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM, pipeline.dtype)
        crossattn_caches_neg = _initialize_crossattn_caches(
            1, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM, pipeline.dtype)

        # --- Block-by-block generation ---
        num_blocks = pipeline.num_frames // pipeline.num_frame_per_block
        current_start_frame = 0
        total_frames_sent = 0
        generation_start = time.time()

        for block_idx in range(num_blocks):
            if stop_event.is_set():
                break

            progress = int(((block_idx + 1) / num_blocks) * 80) + 10
            emit_progress(
                f"Processing block {block_idx + 1}/{num_blocks}...",
                progress, job_id,
            )
            print(f"\n🔄 Block {block_idx + 1}/{num_blocks}")
            block_start = time.time()

            # Get noise for this block
            noisy_input = noise[
                :,
                current_start_frame:
                current_start_frame + pipeline.num_frame_per_block,
            ]

            # --- Denoising loop ---
            denoising_start = time.time()
            for step_idx, current_timestep in enumerate(pipeline.denoising_step_list):
                if stop_event.is_set():
                    break

                timestep = mx.array([current_timestep], dtype=mx.float32)

                # Conditional forward
                flow_pred_cond = pipeline._forward_transformer(
                    noisy_input, timestep, context_cond,
                    kv_caches=kv_caches_pos,
                    crossattn_caches=crossattn_caches_pos,
                )

                # Unconditional forward
                flow_pred_uncond = pipeline._forward_transformer(
                    noisy_input, timestep, context_uncond,
                    kv_caches=kv_caches_neg,
                    crossattn_caches=crossattn_caches_neg,
                )

                # CFG
                flow_pred = flow_pred_uncond + pipeline.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond
                )

                # Euler step
                noisy_input = pipeline.scheduler.step(
                    flow_pred, timestep, noisy_input)

            denoising_time = time.time() - denoising_start
            print(f"  ⚡ Denoising: {denoising_time:.2f}s")

            if stop_event.is_set():
                break

            # --- Update KV cache with clean latents ---
            if block_idx < num_blocks - 1:
                timestep_zero = mx.array([0], dtype=mx.float32)
                pipeline._forward_transformer(
                    noisy_input, timestep_zero, context_cond,
                    kv_caches=kv_caches_pos,
                    crossattn_caches=crossattn_caches_pos,
                )
                pipeline._forward_transformer(
                    noisy_input, timestep_zero, context_uncond,
                    kv_caches=kv_caches_neg,
                    crossattn_caches=crossattn_caches_neg,
                )

            # --- Decode to pixels ---
            decode_start = time.time()
            # (B, T, C, H, W) -> (B, C, T, H, W) for VAE
            vae_input = noisy_input.transpose(0, 2, 1, 3, 4)
            pixels = pipeline.vae.decode(vae_input)
            # Normalize [-1, 1] -> [0, 1]
            pixels = (pixels * 0.5 + 0.5).clip(0, 1)
            decode_time = time.time() - decode_start
            print(f"  🎨 VAE decode: {decode_time:.2f}s")

            # --- Stream frames ---
            block_frames = pixels.shape[2]  # T dimension
            for frame_idx in range(block_frames):
                if stop_event.is_set():
                    break

                # (C, H, W) numpy array
                frame_array = pixels[0, :, frame_idx].astype(mx.float32)
                frame_np = np.array(frame_array)

                # Convert to base64 and send
                b64 = tensor_to_base64_frame(frame_np)
                try:
                    socketio.emit("frame_ready", {
                        "data": b64,
                        "frame_index": total_frames_sent,
                        "block_index": block_idx,
                        "job_id": job_id,
                    })
                except Exception as e:
                    print(f"  ⚠️ Frame send error: {e}")

                # Save frame to disk
                save_frame(frame_np, total_frames_sent)

                total_frames_sent += 1

            block_time = time.time() - block_start
            print(f"  ✅ Block done: {block_time:.2f}s ({block_frames} frames)")

            current_start_frame += pipeline.num_frame_per_block

        # --- Done ---
        generation_time = time.time() - generation_start
        print(f"\n🎉 Generation complete: {generation_time:.1f}s, {total_frames_sent} frames")

        # Generate MP4 from saved frames
        emit_progress("Compiling video...", 95, job_id)
        generate_mp4_from_images(fps=frame_rate)

        emit_progress("Generation complete!", 100, job_id)
        try:
            socketio.emit("generation_complete", {
                "message": "Video generation completed!",
                "total_frames": total_frames_sent,
                "generation_time": f"{generation_time:.1f}s",
                "job_id": job_id,
            })
        except Exception as e:
            print(f"Complete emit error: {e}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            socketio.emit("error", {
                "message": f"Generation failed: {str(e)}",
            })
        except Exception:
            pass
    finally:
        generation_active = False
        stop_event.set()


# ---------------------------------------------------------------------------
# SocketIO event handlers
# ---------------------------------------------------------------------------


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("status", {"message": "Connected to MLX demo server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("start_generation")
def handle_start_generation(data):
    global generation_active, anim_name, frame_rate

    if generation_active:
        emit("error", {"message": "Generation already in progress"})
        return

    prompt = data.get("prompt", "").strip()
    if not prompt:
        emit("error", {"message": "Prompt is required"})
        return

    seed = data.get("seed", -1)
    if seed == -1:
        seed = random.randint(0, 2**32)

    frame_rate = data.get("fps", 6)

    # Create anim name
    words = re.split(r'[^\w\s]', prompt)[0].strip()[:20] if prompt else "video"
    sha = calculate_sha256(prompt)[:10]
    anim_name = f"{words}_{seed}_{sha}"

    print(f"\n🚀 Starting generation: seed={seed}, prompt='{prompt[:60]}...'")

    generation_active = True
    socketio.start_background_task(generate_video_stream, prompt, seed)
    emit("status", {"message": "Generation started"})


@socketio.on("stop_generation")
def handle_stop_generation():
    global generation_active, stop_event
    generation_active = False
    stop_event.set()
    emit("status", {"message": "Generation stopped"})
    print("⏹️ Generation stopped by user")


# ---------------------------------------------------------------------------
# Web routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "generation_active": generation_active,
        "device": "mlx (Apple Silicon)" if hasattr(mx, "metal") and mx.metal is not None else "mlx (CPU)",
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n🚀 Starting MLX demo on http://{args.host}:{args.port}")
    print(f"   Weights: {args.weights}")
    print(f"   Frames: {args.num_frames} x {args.height}x{args.width}")
    print(f"   Blocks: {args.num_frames // args.num_frame_per_block} ({args.num_frame_per_block} frames/block)")
    print()
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
