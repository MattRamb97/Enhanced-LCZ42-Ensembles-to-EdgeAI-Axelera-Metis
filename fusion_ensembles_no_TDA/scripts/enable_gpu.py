import torch

def enable_gpu(idx: int = 0):
    """Select a GPU (like EnableGPU.m) and print info."""
    try:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{idx}")
            gpu_name = torch.cuda.get_device_name(idx)
            free, total = torch.cuda.mem_get_info()
            print(f"[GPU] Using {gpu_name} ({free/1e9:.1f} GB free / {total/1e9:.1f} GB total)")
        else:
            device = torch.device("cpu")
            print("[GPU] No CUDA GPU detected. Using CPU.")
    except Exception as e:
        print(f"[WARN] GPU init failed: {e}. Falling back to CPU.")
        device = torch.device("cpu")
    return device