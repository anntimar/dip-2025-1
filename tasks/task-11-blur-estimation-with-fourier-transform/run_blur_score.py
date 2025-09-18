# run_blur_score.py
import os, sys, argparse, importlib.util
import numpy as np
import cv2

# carrega a função do arquivo com hífen via importlib
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "task-11-blur-estimation-with-fourier-transform.py")
spec = importlib.util.spec_from_file_location("task11_mod", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
frequency_blur_score = mod.frequency_blur_score

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def make_demo_images():
    """Gera uma imagem nítida (xadrez) e uma borrada para teste."""
    H, W, tile = 256, 256, 16
    y, x = np.indices((H, W))
    board = (((x // tile) + (y // tile)) % 2) * 255
    sharp = np.stack([board, board, board], axis=-1).astype(np.uint8)
    blur = cv2.GaussianBlur(sharp, (15, 15), 3.0)
    cv2.imwrite("demo_sharp.png", sharp)
    cv2.imwrite("demo_blur.png", blur)
    return ["demo_sharp.png", "demo_blur.png"]

def main():
    ap = argparse.ArgumentParser(description="Frequency blur score (maior = mais nítida).")
    ap.add_argument("images", nargs="*", help="Caminhos de imagens")
    ap.add_argument("--center", type=int, default=60, help="Tamanho do quadrado central (low-freq).")
    ap.add_argument("--demo", action="store_true", help="Gerar imagens de demonstração e avaliar.")
    args = ap.parse_args()

    paths = args.images
    if args.demo:
        paths = make_demo_images()

    if not paths:
        print("Nenhuma imagem fornecida. Use --demo ou passe caminhos de imagens.")
        sys.exit(1)

    for p in paths:
        img = read_image(p)
        if img is None:
            print(f"[ERRO] não consegui ler: {p}")
            continue
        score = frequency_blur_score(img, center_size=args.center)
        print(f"{os.path.basename(p)}\tcenter={args.center}\tscore={score:.6f}")

if __name__ == "__main__":
    main()
