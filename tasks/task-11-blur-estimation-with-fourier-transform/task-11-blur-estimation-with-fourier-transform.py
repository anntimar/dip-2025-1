"""
task-11-blur-estimation-with-fourier-transform.py

>>> IMPORTANT <<<
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2


def frequency_blur_score(
    image: Union[np.ndarray, "cv2.Mat"],
    center_size: int = 60
) -> float:
    """
    Calcula um escore de nitidez no domínio da frequência.
    Quanto maior o escore, mais nítida a imagem.
    """
    if image is None:
        return 0.0

    # 1) Converte para escala de cinza
    if image.ndim == 3:
        # trata BGR (3 canais) e BGRA (4 canais)
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image

    # 2) Converte para float32 e normaliza suavemente para [0,1] se necessário
    gray = gray.astype(np.float32, copy=False)
    mx = float(gray.max()) if gray.size else 0.0
    if mx > 1.0:  # típico de uint8 em 0..255
        gray = gray / 255.0

    H, W = gray.shape[:2]
    if H == 0 or W == 0:
        return 0.0

    # 3) FFT 2D + shift para centralizar a componente DC
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)

    # 4) Zera um quadrado central (baixas frequências)
    #    Garante que o tamanho do quadrado caiba na imagem
    cs = int(max(1, min(center_size, min(H, W))))
    half = cs // 2
    cy, cx = H // 2, W // 2
    y0 = max(0, cy - half)
    y1 = min(H, cy - half + cs)
    x0 = max(0, cx - half)
    x1 = min(W, cx - half + cs)
    F[y0:y1, x0:x1] = 0

    # 5) Espectro de magnitude (log para compressão dinâmica)
    mag = np.abs(F)
    spec = np.log1p(mag)  # log(1 + |F|)

    # 6) Escore = média do espectro restante (frequências altas)
    score = float(spec.mean()) if spec.size else 0.0
    return score
