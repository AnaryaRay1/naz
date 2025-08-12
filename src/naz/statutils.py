import numpy as np
import pandas as pd

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

def find_level(density, mass=0.9):
    sorted_density = np.sort(density.ravel())[::-1]
    cumsum = np.cumsum(sorted_density)
    cumsum /= cumsum[-1]
    return sorted_density[np.searchsorted(cumsum, mass)]


def equal_quantile_binning_nd(X, n_bins=4, labels=False, return_bin_edges=False):
    """
    Perform equal quantile binning independently for each column of X.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    n_bins : int
        Number of quantile bins per feature.
    labels : bool or list of str
        If True, label bins with integers. If False, return bin indices.
    return_bin_edges : bool
        If True, also return bin edges per feature.

    Returns:
    --------
    X_binned : ndarray of int or category
        Binned version of X.
    bin_edges : list of arrays (if return_bin_edges=True)
        Bin edges used per feature.
    """
    X = np.asarray(X)
    assert X.ndim == 2, "Input must be 2D (n_samples, n_features)"

    X_binned = []
    bin_edges = []

    for i in range(X.shape[1]):
        x_i = X[:, i]
        binned, bins = pd.qcut(x_i, q=n_bins, labels=labels, retbins=True, duplicates='drop')
        X_binned.append(binned.codes if hasattr(binned, 'codes') else binned)
        bin_edges.append(bins)

    X_binned = np.vstack(X_binned).T
    return (X_binned, bin_edges) if return_bin_edges else X_binned

