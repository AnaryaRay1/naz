import jax
_ = jax.devices()

import torch
import numpy as np

def set_device(tensor,device="cuda", dtype=torch.float32):
    """
    Convert list or numpy array to torch tensor on particular device
    """
    if device == "cpu":
        pass
    elif device=="cuda":
        current_gpu_index = torch.cuda.current_device()
        device = f"cuda:{current_gpu_index}"
        torch.cuda.set_device(device)
    else:
        raise
    return torch.as_tensor(
                    tensor, dtype=dtype, device=torch.device(device)
                )

device = set_device([0]).device
