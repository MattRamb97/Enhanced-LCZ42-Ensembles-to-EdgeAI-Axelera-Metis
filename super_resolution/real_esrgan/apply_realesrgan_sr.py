import h5py
import numpy as np
import torch
from tqdm import tqdm
import os
import gc
import sys
import types


def ensure_torchvision_stub() -> None:
    """Provide minimal torchvision.transforms.functional_tensor.rgb_to_grayscale."""
    module_name = "torchvision.transforms.functional_tensor"
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except Exception:
        pass

    def rgb_to_grayscale(img: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
        if img.dim() not in (3, 4):
            raise ValueError("Expected 3D or 4D tensor")
        channel_dim = -3
        if img.size(channel_dim) != 3:
            raise ValueError("Input tensor must have 3 channels")
        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img.dtype, device=img.device)
        view_shape = [1] * img.dim()
        view_shape[channel_dim] = 3
        gray = (img * weights.view(*view_shape)).sum(dim=channel_dim, keepdim=True)
        if num_output_channels == 3:
            gray = gray.repeat_interleave(3, dim=channel_dim)
        elif num_output_channels != 1:
            raise ValueError("num_output_channels must be 1 or 3")
        return gray

    tv_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_functional_module = types.ModuleType("torchvision.transforms.functional")
    utils_module = types.ModuleType("torchvision.utils")
    models_module = types.ModuleType("torchvision.models")
    vgg_module = types.ModuleType("torchvision.models.vgg")

    def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False):
        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in tensor])
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError("make_grid expects a 4D tensor")
        return tensor

    functional_tensor_module = types.ModuleType(module_name)
    functional_tensor_module.rgb_to_grayscale = rgb_to_grayscale
    utils_module.make_grid = make_grid

    class _DummyVGG(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            return x

    def vgg16(*args, **kwargs):
        return _DummyVGG()

    vgg_module.vgg16 = vgg16
    models_module.vgg = vgg_module

    sys.modules.setdefault("torchvision", tv_module)
    sys.modules.setdefault("torchvision.transforms", transforms_module)
    sys.modules.setdefault("torchvision.transforms.functional", transforms_functional_module)
    sys.modules.setdefault("torchvision.utils", utils_module)
    sys.modules.setdefault("torchvision.models", models_module)
    sys.modules.setdefault("torchvision.models.vgg", vgg_module)
    sys.modules[module_name] = functional_tensor_module
    def normalize(tensor, mean, std, inplace=False):
        tensor = torch.as_tensor(tensor)
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        if inplace:
            tensor.sub_(mean[..., None, None]).div_(std[..., None, None])
            return tensor
        return (tensor - mean[..., None, None]) / std[..., None, None]

    transforms_functional_module.normalize = normalize

    transforms_module.functional = transforms_functional_module
    tv_module.transforms = transforms_module
    tv_module.utils = utils_module
    tv_module.models = models_module


ensure_torchvision_stub()

from basicsr.archs.rrdbnet_arch import RRDBNet

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
EPS = 1e-6

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
SCALE = 4  # can also set to 2
MODEL_PATH = f"RealESRGAN_x{SCALE}plus.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

INPUT_FILES = [
    os.path.join(DATA_DIR, "training.h5"),
    os.path.join(DATA_DIR, "testing.h5")
]

# ---------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------
def load_rrdb_model() -> torch.nn.Module:
    print(f"[INFO] Loading pretrained {MODEL_PATH} on {DEVICE} ...")
    net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=SCALE,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["params_ema", "params", "state_dict", "model"]:
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    net.load_state_dict(checkpoint, strict=True)
    net.to(DEVICE).eval()
    torch.set_grad_enabled(False)
    print("[INFO] Model loaded successfully.")
    return net


model = load_rrdb_model()


def run_rrdb(batch_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        sr = model(batch_tensor.to(DEVICE))
    return sr.cpu().clamp(0, 1).numpy()


def calibrate_stack(sr_stack: np.ndarray, lr_stack: np.ndarray, scale: int) -> np.ndarray:
    B, C, Hs, Ws = sr_stack.shape
    H, W = lr_stack.shape[2:]
    sr_down = sr_stack.reshape(B, C, H, scale, W, scale).mean(axis=(3, 5))
    gains = lr_stack.std(axis=(2, 3), keepdims=True) / (sr_down.std(axis=(2, 3), keepdims=True) + EPS)
    biases = lr_stack.mean(axis=(2, 3), keepdims=True) - gains * sr_down.mean(axis=(2, 3), keepdims=True)
    sr_corrected = sr_stack * gains + biases
    return np.clip(sr_corrected, 0.0, 2.8)

# ---------------------------------------------------------------
# Process datasets
# ---------------------------------------------------------------
for input_h5 in INPUT_FILES:
    base = os.path.splitext(os.path.basename(input_h5))[0]
    output_h5 = os.path.join(DATA_DIR, f"{base}_realesrgan{SCALE}x.h5")

    with h5py.File(input_h5, "r") as f_in:
        N, H, W, C = f_in["/sen2"].shape
        labels = f_in["/label"][:]
        print(f"\n[INFO] Processing {base}.h5 → {base}_realesrgan{SCALE}x.h5")
        print(f"       Found {N} patches ({H}×{W}×{C})")

    with h5py.File(output_h5, "w") as f_out:
        f_out.create_dataset("sen2", shape=(N, H*SCALE, W*SCALE, C), dtype="float16")
        f_out.create_dataset("label", data=labels, dtype="uint8")

        with h5py.File(input_h5, "r") as f_in:
            for start in tqdm(range(0, N, BATCH_SIZE), desc=f"{base} ×{SCALE}"):
                end = min(start + BATCH_SIZE, N)
                patch_batch = f_in["/sen2"][start:end].astype(np.float32)  # (B,H,W,C)
                B = patch_batch.shape[0]

                norm = np.clip(patch_batch / 2.8, 0, 1)
                bands = norm.transpose(0, 3, 1, 2).reshape(-1, 1, H, W)
                inputs = np.repeat(bands, 3, axis=1)
                sr = run_rrdb(torch.from_numpy(inputs))
                sr_gray = np.tensordot(sr, LUMA_WEIGHTS, axes=([1], [0])) * 2.8
                sr_gray = sr_gray.reshape(B, C, H * SCALE, W * SCALE)
                lr_stack = patch_batch.transpose(0, 3, 1, 2)
                sr_gray = calibrate_stack(sr_gray, lr_stack, SCALE)
                sr_gray = sr_gray.transpose(0, 2, 3, 1).astype(np.float16)
                f_out["/sen2"][start:end] = sr_gray

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    print(f"Saved {output_h5}")

print("\nReal-ESRGAN processing completed successfully for all datasets.")
