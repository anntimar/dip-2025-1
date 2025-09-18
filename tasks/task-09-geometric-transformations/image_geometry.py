# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    """
    Aplica transformações geométricas em uma imagem 2D (tons de cinza) usando apenas NumPy.
    Áreas mapeadas para fora da imagem original são preenchidas com 0.
    """
    if img.ndim != 2:
        raise ValueError("A entrada deve ser uma imagem 2D (tons de cinza).")

    H, W = img.shape
    img_dtype = img.dtype

    # 1) Translação: deslocar para a direita e para baixo (10% do tamanho)
    ty = max(1, int(0.10 * H))  # deslocamento vertical
    tx = max(1, int(0.10 * W))  # deslocamento horizontal
    translated = np.zeros_like(img)
    translated[ty:H, tx:W] = img[0:H - ty, 0:W - tx]

    # 2) Rotação de 90 graus no sentido horário
    # Transpor e depois inverter colunas
    rotated = img.T[:, ::-1].copy()

    # 3) Esticamento horizontal (escala na largura por 1.5)
    scale = 1.5
    new_W = max(1, int(round(W * scale)))
    stretched = np.zeros((H, new_W), dtype=img_dtype)
    # Amostragem por vizinho mais próximo via mapeamento inverso: x_in = x_out / scale
    x_out = np.arange(new_W)
    x_in = np.round(x_out / scale).astype(int)
    x_in = np.clip(x_in, 0, W - 1)
    stretched[:] = img[:, x_in]

    # 4) Espelhamento horizontal (inverter ao longo do eixo vertical)
    mirrored = img[:, ::-1].copy()

    # 5) Distorção do tipo "barrel" (distorção radial simples)
    # Normaliza coordenadas em relação ao centro da imagem para o intervalo [-1, 1]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy

    # Parâmetro de intensidade da distorção (k > 0 -> barrel)
    k = 0.3

    r2 = xn**2 + yn**2
    denom = (1.0 + k * r2)

    # Mapeamento inverso para amostrar da imagem original
    xs = xn / denom
    ys = yn / denom

    # Converte de volta para coordenadas de pixels da imagem de origem
    x_src = xs * cx + cx
    y_src = ys * cy + cy

    # Amostragem por vizinho mais próximo com checagem de limites
    x_nn = np.round(x_src).astype(int)
    y_nn = np.round(y_src).astype(int)
    inside = (x_nn >= 0) & (x_nn < W) & (y_nn >= 0) & (y_nn < H)

    distorted = np.zeros_like(img)
    distorted[inside] = img[y_nn[inside], x_nn[inside]]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted,
    }
