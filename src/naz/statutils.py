import numpy as np

def hpd(samples,alpha = 0.1):
    x=np.sort(np.copy(samples))
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]

    return [hdi_min, hdi_max]

def hpd_vectorized(samples, alpha=0.1):
    """
    Vectorized HPD for samples with shape (ns, nx, ny), returns (2, nx, ny)
    """
    # Sort along the sample axis (axis=0)
    x = np.sort(samples, axis=0)
    ns = x.shape[0]
    cred_mass = 1.0 - alpha
    interval_idx_inc = int(np.floor(cred_mass * ns))
    n_intervals = ns - interval_idx_inc

    if n_intervals <= 0:
        raise ValueError("Too few elements for interval calculation")

    # Compute interval widths: shape (n_intervals, nx, ny)
    interval_width = x[interval_idx_inc:, ...] - x[:n_intervals, ...]

    # Find the index with the smallest width along axis 0
    min_idx = np.argmin(interval_width, axis=0)  # shape (nx, ny)

    # Prepare an output array of shape (2, nx, ny)
    hdi_min = np.take_along_axis(x, min_idx[None, :, :], axis=0)[0]
    hdi_max = np.take_along_axis(x, (min_idx + interval_idx_inc)[None, :, :], axis=0)[0]

    return np.stack([hdi_min, hdi_max], axis=0)

