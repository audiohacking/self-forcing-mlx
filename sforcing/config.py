# 1.3B model configuration constants
# Extracted from wan/configs/wan_t2v_1_3B.py and shared_config.py


# Transformer
DIM = 1536
FFN_DIM = 8960
NUM_HEADS = 12
NUM_LAYERS = 30
HEAD_DIM = DIM // NUM_HEADS  # 128
FREQ_DIM = 256
WINDOW_SIZE = (-1, -1)  # global attention
QK_NORM = True
CROSS_ATTN_NORM = True
EPS = 1e-6
PATCH_SIZE = (1, 2, 2)

# Text / T5
TEXT_DIM = 4096
TEXT_LEN = 512
T5_VOCAB_SIZE = 256384
T5_DIM = 4096
T5_DIM_ATTN = 4096
T5_DIM_FFN = 10240
T5_NUM_HEADS = 64
T5_ENCODER_LAYERS = 24
T5_NUM_BUCKETS = 32
T5_SHARED_POS = False
T5_DROPOUT = 0.1

# Input / Output
IN_DIM = 16
OUT_DIM = 16

# Sequence lengths
FRAME_SEQ_LEN = 1560  # (21//4) * (480//8) * (832//8) = 5 * 60 * 104 = 31200 for 21 frames
# For 1 frame: 1 * 60 * 104 = 6240 tokens per frame
MAX_SEQ_LEN = 32760  # max cache size

# VAE
VAE_Z_DIM = 16
VAE_DIM = 96
VAE_DIM_MULT = [1, 2, 4, 4]
VAE_NUM_RES_BLOCKS = 2
VAE_ATTN_SCALES = []
VAE_TEMPORAL_DOWNSAMPLE = [False, True, True]
VAE_TEMPORAL_UPSAMPLE = [True, True, False]  # reverse of downsample

# Scheduler
NUM_TRAIN_TIMESTEPS = 1000

# Inference defaults (1.3B)
TIMESTEP_SHIFT = 5.0
GUIDANCE_SCALE = 3.0
DENOISING_STEP_LIST = [1000, 750, 500, 250]
NUM_FRAME_PER_BLOCK = 3

# Memory
SAMPLE_FPS = 16

# bf16 support
BF16_ENABLED = True