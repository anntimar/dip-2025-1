# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """
    Ajusta, canal a canal (R, G, B), o histograma da imagem fonte para
    coincidir com o histograma da imagem de referência.
    Retorna a imagem resultante em uint8.
    """
    if source_img.ndim != 3 or source_img.shape[2] != 3:
        raise ValueError("source_img deve ter formato (H, W, 3) em RGB.")
    if reference_img.ndim != 3 or reference_img.shape[2] != 3:
        raise ValueError("reference_img deve ter formato (H, W, 3) em RGB.")

    # Garantir tipo uint8 no retorno; trabalhar internamente com 0..255
    src = source_img.astype(np.uint8, copy=False)
    ref = reference_img.astype(np.uint8, copy=False)

    matched = np.empty_like(src)

    # Função auxiliar: cria LUT de mapeamento pela correspondência de CDFs
    def build_lut(src_channel: np.ndarray, ref_channel: np.ndarray) -> np.ndarray:
        # histogramas com 256 níveis (0..255)
        hist_src = np.bincount(src_channel.ravel(), minlength=256).astype(np.float64)
        hist_ref = np.bincount(ref_channel.ravel(), minlength=256).astype(np.float64)

        # CDFs normalizadas
        cdf_src = np.cumsum(hist_src)
        cdf_ref = np.cumsum(hist_ref)
        if cdf_src[-1] == 0 or cdf_ref[-1] == 0:
            # caso degenerado (imagem vazia); mapeia identidade
            return np.arange(256, dtype=np.uint8)

        cdf_src /= cdf_src[-1]
        cdf_ref /= cdf_ref[-1]

        # Para cada quantil da fonte, encontra intensidade na ref com CDF >= quantil
        # Usamos interpolação contínua entre níveis 0..255
        levels = np.arange(256)
        lut = np.interp(cdf_src, cdf_ref, levels)

        # Arredonda e limita para 0..255
        lut = np.clip(np.rint(lut), 0, 255).astype(np.uint8)
        return lut

    # Aplica canal a canal
    for c in range(3):
        lut = build_lut(src[..., c], ref[..., c])
        matched[..., c] = lut[src[..., c]]

    return matched.astype(np.uint8, copy=False)
