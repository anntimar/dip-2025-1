# run_histogram_match.py
import os, importlib.util
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "task-07-histogram-matching.py")  # arquivo com hífen

spec = importlib.util.spec_from_file_location("task07_module", SRC_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# pega só a função que precisamos
match_histograms_rgb = mod.match_histograms_rgb
def read_rgb_uint8(path):
    img = mpimg.imread(path)
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        # se vier como float [0,1], escalar para [0,255]
        if img.max() <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img[..., :3]

def main():
    src = read_rgb_uint8("source.jpg")
    ref = read_rgb_uint8("reference.jpg")
    out = match_histograms_rgb(src, ref)
    plt.imsave("output.jpg", out)  # salva em RGB
    print("Gerado: output.jpg")

if __name__ == "__main__":
    main()
