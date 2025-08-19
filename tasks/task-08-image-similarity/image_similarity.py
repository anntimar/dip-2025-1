# image_similarity_solution.py
# REFERENCE SOLUTION — FOR INSTRUCTOR USE
from __future__ import annotations
import numpy as np


def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    _validate_images(i1, i2)

    # Cast to float64 for numerical stability across all metrics
    i1 = i1.astype(np.float64, copy=False)
    i2 = i2.astype(np.float64, copy=False)

    mse_val = _mse(i1, i2)
    psnr_val = _psnr(i1, i2, data_range=1.0)
    ssim_val = _ssim(i1, i2, C1=1e-8, C2=1e-8)
    npcc_val = _npcc(i1, i2)

    return {
        "mse": float(mse_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "npcc": float(npcc_val),
    }


# ----------------------------- Helpers --------------------------------- #

def _validate_images(i1: np.ndarray, i2: np.ndarray) -> None:
    if not (isinstance(i1, np.ndarray) and isinstance(i2, np.ndarray)):
        raise ValueError("Inputs must be NumPy arrays.")
    if i1.ndim != 2 or i2.ndim != 2:
        raise ValueError("Inputs must be 2D arrays representing grayscale images.")
    if i1.shape != i2.shape:
        raise ValueError(f"Images must have the same shape. Got {i1.shape} vs {i2.shape}.")


def _mse(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    MSE = mean( (i1 - i2)^2 )
    """
    ### START CODE HERE ###
    mse = np.mean((i1 - i2) ** 2)
    ### END CODE HERE ###
    return mse


def _psnr(i1: np.ndarray, i2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    PSNR = 10 * log10( (L^2) / MSE ), where L = data_range (peak value).
    Returns np.inf if mse == 0.
    """
    ### START CODE HERE ###
    mse = _mse(i1, i2)
    if mse == 0:
        psnr = np.inf
    else:
        psnr = 10.0 * np.log10((data_range ** 2) / mse)
    ### END CODE HERE ###
    return psnr


def _ssim(i1: np.ndarray, i2: np.ndarray, *, C1: float = 1e-8, C2: float = 1e-8) -> float:
    """
    Simplified global Structural Similarity Index (SSIM).
    """
    ### START CODE HERE ###
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)

    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12   = np.mean((i1 - mu1) * (i2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    ### END CODE HERE ###
    return ssim


def _npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Normalized Pearson Correlation Coefficient (NPCC).
    """
    ### START CODE HERE ###
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)

    num = np.sum((i1 - mu1) * (i2 - mu2))
    denom = np.sqrt(np.sum((i1 - mu1) ** 2) * np.sum((i2 - mu2) ** 2))

    # If denominator is zero and images are identical constant, return 1.0.
    if denom == 0:
        if np.allclose(i1, i2):
            npcc = 1.0
        else:
            npcc = 0.0
    else:
        npcc = num / denom
    ### END CODE HERE ###
    return npcc


# -------------------------- Self-check (optional) ----------------------- #

if __name__ == "__main__":
    i1 = np.array([[0.0, 0.5],
                   [0.5, 1.0]], dtype=np.float64)
    i2 = np.array([[0.0, 0.4],
                   [0.6, 1.0]], dtype=np.float64)

    out = compare_images(i1, i2)
    pretty = {
        "mse": round(out["mse"], 6),
        "psnr": round(out["psnr"], 2) if np.isfinite(out["psnr"]) else float("inf"),
        "ssim": round(out["ssim"], 6),
        "npcc": round(out["npcc"], 6),
    }
    print(pretty)
