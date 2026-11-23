import numpy as np, cv2
from pathlib import Path

SRC = Path(r"D:\EmbeddedSystems_E26\cnn_repo\shapes_aug_test_txt")   # pixel .txt files
DST = Path(r"D:\EmbeddedSystems_E26\cnn_repo\snn_test_features")      # output
CLASSES = ["square", "circle", "figure8", "triangle"]                 # label = index
T, GRID_H, GRID_W = 5, 12, 12
PERCENTILE, ALPHA, SCALE = 90, 3.0, 128
MAX_PER_CLASS = None       # evenly spaced files per class (None = all)

def image_to_spikes(img_u8):
    H, W = img_u8.shape
    img = img_u8.astype(np.float32) / 255.0
    s = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    pos, neg = np.maximum(s, 0), np.maximum(-s, 0)
    pos /= pos.max() + 1e-8
    neg /= neg.max() + 1e-8
    on_mask = pos > np.percentile(pos, PERCENTILE)
    off_mask = neg > np.percentile(neg, PERCENTILE)
    spikes = np.zeros((T, 2, H, W), dtype=np.float32)
    on_t = (np.exp(-ALPHA * pos) * (T - 1)).astype(np.int32)
    off_t = (np.exp(-ALPHA * neg) * (T - 1)).astype(np.int32)
    ys, xs = np.where(on_mask);  spikes[on_t[ys, xs], 0, ys, xs] = 1.0
    ys, xs = np.where(off_mask); spikes[off_t[ys, xs], 1, ys, xs] = 1.0
    return spikes

def spikes_to_features(spikes):
    _, _, H, W = spikes.shape
    ch, cw = H // GRID_H, W // GRID_W
    on = spikes[:, 0].reshape(T, GRID_H, ch, GRID_W, cw).mean(axis=(2, 4))
    off = spikes[:, 1].reshape(T, GRID_H, ch, GRID_W, cw).mean(axis=(2, 4))
    f = np.stack([on, off], axis=-1).reshape(T, GRID_H * GRID_W * 2)   # (T, 288)
    return np.round(f * SCALE).astype(np.int16).reshape(-1)            # 1440, t-major

for label, cname in enumerate(CLASSES):
    files = sorted((SRC / cname).glob("*.txt"))
    if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
        idx = np.linspace(0, len(files) - 1, MAX_PER_CLASS).astype(int)
        files = [files[i] for i in idx]
    out_dir = DST / cname
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        img = np.loadtxt(p, dtype=np.float32).astype(np.uint8)         # 60x60
        assert img.shape == (60, 60), (p, img.shape)
        feats = spikes_to_features(image_to_spikes(img))
        assert feats.size == T * GRID_H * GRID_W * 2
        with open(out_dir / p.name, "w") as f:
            f.write(f"# label {label}\n# image {p}\n# shape T={T}, N_FEATURES={GRID_H*GRID_W*2}\n")
            f.write("\n".join(str(int(v)) for v in feats) + "\n")
    print(cname, len(files), "files")
