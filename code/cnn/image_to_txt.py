from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_aug_test_60x60")
OUTPUT_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_aug_test_txt")

INPUT_FOLDERS = [
    "square",    # 1
    "circle",    # 2
    "figure8",   # 3
    "triangle",  # 4
]

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def image_to_txt(img_path: Path, out_path: Path):
    img = Image.open(img_path).convert("L")      # 8-bit grayscale
    if img.size != (60, 60):
        img = img.resize((60, 60), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)        # shape (60, 60)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in arr:
            f.write(" ".join(str(int(v)) for v in row) + "\n")

for folder in INPUT_FOLDERS:
    in_dir = DATA_ROOT / folder
    out_dir = OUTPUT_ROOT / folder
    for p in sorted(in_dir.iterdir()):
        if p.suffix.lower() in EXTS:
            image_to_txt(p, out_dir / (p.stem + ".txt"))
            print("wrote", out_dir / (p.stem + ".txt"))

