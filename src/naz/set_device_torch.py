import torch
import numpy as np

def set_device(tensor, device=None, dtype=torch.float32):
    """
    Convert a list or numpy array to a torch tensor on a specified device.
    Accepts device as a string (e.g., "cuda:0", "cpu") or a torch.device object.
    """
    # Default to CUDA if available
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if isinstance(device, str):
        dev = torch.device(device)
    elif isinstance(device, torch.device):
        dev = device
    else:
        raise ValueError(f"Unknown device type: {device}")

    # Just move tensor to device; no need to call torch.cuda.set_device()
    return torch.as_tensor(tensor, dtype=dtype, device=dev)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
