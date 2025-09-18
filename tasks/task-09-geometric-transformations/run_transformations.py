# run_transformations.py
import numpy as np
import matplotlib.pyplot as plt
from image_geometry import apply_geometric_transformations

def load_or_make_test_image():
    """
    Tenta carregar uma imagem 'input.png' (ou .jpg), convertendo para tons de cinza.
    Se não existir, gera uma imagem sintética (quadriculado) 240x320.
    """
    import os
    import matplotlib.image as mpimg

    # tenta carregar um arquivo de imagem se existir
    for name in ["input.png", "input.jpg", "input.jpeg", "input.bmp"]:
        if os.path.exists(name):
            img = mpimg.imread(name)
            # converte para grayscale (2D)
            if img.ndim == 3:
                # se tiver alpha, ignora
                img = img[..., :3]
                # média dos canais RGB
                img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
            # normaliza para [0,1]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            return img

    # gera um quadriculado sintético (fallback)
    H, W = 240, 320
    y, x = np.indices((H, W))
    tile = ((x // 20) + (y // 20)) % 2
    img = tile.astype(np.float32)
    return img

def save_gray_png(path, arr):
    """
    Salva um array 2D em PNG (grayscale) usando matplotlib.
    """
    arr = np.asarray(arr)
    # clipe e normaliza para [0,1] caso necessário
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    if arr.min() < 0 or arr.max() > 1:
        # normaliza simples (evita divisão por zero)
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn + 1e-9)
    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    img = load_or_make_test_image()
    out = apply_geometric_transformations(img)

    # salva resultados
    save_gray_png("00_input.png", img)
    save_gray_png("01_translated.png", out["translated"])
    save_gray_png("02_rotated.png", out["rotated"])
    save_gray_png("03_stretched.png", out["stretched"])
    save_gray_png("04_mirrored.png", out["mirrored"])
    save_gray_png("05_distorted.png", out["distorted"])

    print("Arquivos salvos:")
    print("  00_input.png")
    print("  01_translated.png")
    print("  02_rotated.png")
    print("  03_stretched.png")
    print("  04_mirrored.png")
    print("  05_distorted.png")

if __name__ == "__main__":
    main()
