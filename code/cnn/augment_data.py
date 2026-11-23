# #!/usr/bin/env python3
# from pathlib import Path
# import cv2
# import albumentations as A

# # ===================== EDIT THESE =====================
# # DATA_ROOT = Path("/home/dinesh/drone_training_data")
# DATA_ROOT = Path("/home/dinesh/recordings/new_testing_data")

# INPUT_FOLDERS = [
#     "Full Cable",
#     "Left End",
#     "Right End",
#     "No Cable",
# ]

# # Output folder name (as requested)
# # OUTPUT_ROOT = DATA_ROOT.parent / "drone_training_data_aug"
# OUTPUT_ROOT = Path("/home/dinesh/recordings/new_testing_data_aug")

# # Options
# SAVE_ORIGINALS = True          # save *_orig files too
# OVERWRITE      = False         # if False, skip writing if output already exists
# WRITE_README   = True          # write README_augmentations.txt

# # Apply JPEGCompression ONLY to a subset of images in ONE folder
# JPEG_ONLY_FOLDER = "drone_cable_right_end"
# JPEG_ONLY_COUNT  = 0
# # ======================================================

# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# def is_image_file(p: Path) -> bool:
#     return p.is_file() and p.suffix.lower() in IMG_EXTS


# # ----------------- Augmentations -----------------
# BORDER = cv2.BORDER_REFLECT_101

# aug_lighting = A.Compose([
#     A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
#     A.RandomGamma(gamma_limit=(80, 120), p=1.0),
# ])

# aug_gaussian     = A.Compose([A.GaussianBlur(blur_limit=(3, 5), p=1.0)])
# aug_motion_blur  = A.Compose([A.MotionBlur(blur_limit=7, p=1.0)])
# aug_fog          = A.Compose([A.RandomFog(p=1.0, fog_coef_range=(0.12, 0.32), alpha_coef=0.06)])
# aug_optical      = A.Compose([A.OpticalDistortion(distort_limit=0.1, p=1.0)])
# aug_clahe        = A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)])
# aug_sharpen      = A.Compose([A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=1.0)])
# aug_median       = A.Compose([A.MedianBlur(blur_limit=5, p=1.0)])
# aug_defocus      = A.Compose([A.Defocus(radius=(3, 6), alias_blur=(0.2, 0.5), p=1.0)])
# aug_iso_noise    = A.Compose([A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0)])
# aug_hsv          = A.Compose([A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0)])
# aug_tonecurve    = A.Compose([A.RandomToneCurve(scale=0.1, p=1.0)])
# aug_gamma_only   = A.Compose([A.RandomGamma(gamma_limit=(80, 120), p=1.0)])
# aug_downscale    = A.Compose([A.Downscale(scale_range=(0.65, 0.88), p=1.0)])
# aug_multnoise    = A.Compose([A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0)])
# aug_posterize    = A.Compose([A.Posterize(num_bits=4, p=1.0)])

# # JPEG compression (extra augmentation)
# aug_jpeg = A.Compose([A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0)])

# # Geometric / drone jitter
# aug_roi_jitter = A.Compose([
#     A.CropAndPad(percent=(-0.04, 0.04), border_mode=BORDER, keep_size=True, p=1.0)
# ])

# aug_mild_perspective = A.Compose([
#     A.Perspective(scale=(0.01, 0.04), keep_size=True, border_mode=BORDER, fit_output=False, p=1.0)
# ])

# aug_affine_mild = A.Compose([
#     A.Affine(
#         scale=(0.97, 1.03),
#         translate_percent=(-0.03, 0.03),
#         rotate=(-3, 3),
#         shear=(-2, 2),
#         fit_output=False,
#         border_mode=BORDER,
#         p=1.0
#     )
# ])

# aug_safe_rotate = A.Compose([
#     A.SafeRotate(limit=4, border_mode=BORDER, p=1.0)
# ])

# AUGMENTATIONS = [
#     ("LightingOnly",      aug_lighting),
#     ("GaussianBlur",      aug_gaussian),
#     ("MotionBlur",        aug_motion_blur),
#     ("FogHaze",           aug_fog),
#     ("OpticalDistortion", aug_optical),
#     ("CLAHE",             aug_clahe),
#     ("Sharpen",           aug_sharpen),
#     ("MedianBlur",        aug_median),
#     ("Defocus",           aug_defocus),
#     ("ISONoise",          aug_iso_noise),
#     ("HueSatValue",       aug_hsv),
#     ("ToneCurve",         aug_tonecurve),
#     ("GammaOnly",         aug_gamma_only),
#     ("Downscale",         aug_downscale),
#     ("Multiplicative",    aug_multnoise),
#     ("Posterize",         aug_posterize),
#     ("roijitter",         aug_roi_jitter),
#     ("mildperspective",   aug_mild_perspective),
#     ("affinemild",        aug_affine_mild),
#     ("saferotate",        aug_safe_rotate),
# ]


# def maybe_write(path: Path, img):
#     if (not OVERWRITE) and path.exists():
#         return
#     path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(path), img)


# def write_readme():
#     if not WRITE_README:
#         return
#     OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
#     readme = OUTPUT_ROOT / "README_augmentations.txt"
#     lines = [
#         "Augmentation pipeline\n\n",
#         "Total augmentations per image: {}\n".format(len(AUGMENTATIONS)),
#         "Original image saved: {}\n\n".format(SAVE_ORIGINALS),
#         "Augmentations:\n",
#     ]
#     for i, (name, _) in enumerate(AUGMENTATIONS, 1):
#         lines.append(f"  {i:02d}. {name}\n")

#     lines.append("\nExtra augmentation (subset only):\n")
#     lines.append(f"  - JPEGCompression applied to first {JPEG_ONLY_COUNT} images in folder '{JPEG_ONLY_FOLDER}'\n")
#     lines.append("    (saved as *_augEX_JPEGCompression.<ext>)\n")

#     readme.write_text("".join(lines), encoding="utf-8")


# def main():
#     print("[INFO] Input root :", DATA_ROOT)
#     print("[INFO] Output root:", OUTPUT_ROOT)
#     print("[INFO] Classes    :", INPUT_FOLDERS)
#     print("[INFO] Augmentations per image:", len(AUGMENTATIONS))
#     print(f"[INFO] Extra JPEGCompression: folder={JPEG_ONLY_FOLDER} count={JPEG_ONLY_COUNT}")
#     print()

#     write_readme()

#     for folder in INPUT_FOLDERS:
#         in_dir = DATA_ROOT / folder
#         if not in_dir.exists():
#             print(f"[WARN] Missing folder: {in_dir}")
#             continue

#         out_dir = OUTPUT_ROOT / folder
#         out_dir.mkdir(parents=True, exist_ok=True)

#         files = sorted(p for p in in_dir.rglob("*") if is_image_file(p))
#         print(f"[FOLDER] {folder}: {len(files)} images")

#         for i, p in enumerate(files, start=1):
#             img = cv2.imread(str(p), cv2.IMREAD_COLOR)
#             if img is None:
#                 continue

#             rel  = p.relative_to(in_dir)
#             base = rel.stem
#             ext  = rel.suffix

#             if SAVE_ORIGINALS:
#                 maybe_write(out_dir / rel.parent / f"{base}_orig{ext}", img)

#             for idx, (name, aug) in enumerate(AUGMENTATIONS, 1):
#                 try:
#                     aug_img = aug(image=img)["image"]
#                 except Exception as e:
#                     print(f"[ERR] {name} failed on {p.name}: {e}")
#                     continue

#                 out_name = f"{base}_aug{idx:02d}_{name}{ext}"
#                 maybe_write(out_dir / rel.parent / out_name, aug_img)

#             # ----- Extra JPEGCompression on a subset of ONE folder -----
#             if folder == JPEG_ONLY_FOLDER and i <= JPEG_ONLY_COUNT:
#                 try:
#                     jpeg_img = aug_jpeg(image=img)["image"]
#                     out_name = f"{base}_augEX_JPEGCompression{ext}"
#                     maybe_write(out_dir / rel.parent / out_name, jpeg_img)
#                 except Exception as e:
#                     print(f"[ERR] JPEGCompression failed on {p.name}: {e}")
#             # ----------------------------------------------------------

#             if i % 200 == 0:
#                 print(f"  ... {i}/{len(files)} done")

#     print("\n[DONE] Augmented dataset written to:")
#     print(" ", OUTPUT_ROOT)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
from pathlib import Path
import math
import cv2
import numpy as np

# ===================== EDIT THESE =====================
DATA_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_test_60x60")
# DATA_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_dataset_60x60")
INPUT_FOLDERS = [
    "square",       #1
    "circle",       #2
    "figure8",      #3
    "triangle",     #4
]

OUTPUT_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_aug_test_60x60")
# OUTPUT_ROOT = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync/shapes_aug_dataset_60x60")

# Options
SAVE_ORIGINALS = False          # save *_orig files too
OVERWRITE      = False         # if False, skip writing if output already exists
WRITE_README   = True          # write README_augmentations.txt
SEED           = None          # set an int (e.g. 42) for reproducible augmentations

# Augmentation parameters
BLUR_KSIZES      = (3, 5)        # Gaussian kernel sizes to pick from
CONTRAST_RANGE   = (0.7, 1.3)    # contrast factor around image mean
COMPRESS_RANGE   = (0.70, 0.90)  # width scale for compression
SHADE_STRENGTH   = (0.30, 0.60)  # how much darker the darkest side gets
NOISE_SIGMA      = (8, 20)       # white-noise std dev (pixel units)
USE_FLIPS        = False         # True: rotation aug may also pick h/v flip

# Collage preview: original + all augmentations in one figure
MAKE_COLLAGE        = True     # set False to turn off
COLLAGE_PER_FOLDER  = 2        # how many images per class get a collage
COLLAGE_COLS        = 6        # grid columns (11 tiles -> 6x2 grid)
COLLAGE_SCALE       = 3        # upscale 60x60 tiles to 180x180 so they're visible
COLLAGE_DIR         = OUTPUT_ROOT / "_collages"
# ======================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
rng = np.random.default_rng(SEED)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


# ----------------- Helpers -----------------
def background_color(img):
    """Estimate background colour as the median of the outer 1-px frame."""
    frame = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
    return np.median(frame, axis=0).astype(np.uint8)


def to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


# ----------------- Single augmentations -----------------
def gaussian_blur(img):
    k = int(rng.choice(BLUR_KSIZES))
    return cv2.GaussianBlur(img, (k, k), 0)


def contrast(img):
    a = rng.uniform(*CONTRAST_RANGE)
    mean = img.reshape(-1, img.shape[2]).mean(axis=0)
    return to_uint8((img.astype(np.float32) - mean) * a + mean)


def compress(img):
    """Squeeze horizontally, centre it, pad with background colour (no reflection)."""
    h, w = img.shape[:2]
    s = rng.uniform(*COMPRESS_RANGE)
    nw = max(1, int(round(w * s)))
    small = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)
    out = np.empty_like(img)
    out[:] = background_color(img)
    x0 = (w - nw) // 2
    out[:, x0:x0 + nw] = small
    return out


def gradient_shade(img):
    """Linear shading in a random direction: one side darker (e.g. only top-right)."""
    h, w = img.shape[:2]
    theta = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xs - w / 2) * np.cos(theta) + (ys - h / 2) * np.sin(theta)
    t = (proj - proj.min()) / max(np.ptp(proj), 1e-6)    # 0 .. 1 across the image
    factor = 1.0 - rng.uniform(*SHADE_STRENGTH) * t       # 1 = untouched, lower = darker
    return to_uint8(img.astype(np.float32) * factor[..., None])


def white_noise(img):
    """Additive Gaussian white noise on every pixel (object AND background)."""
    sigma = rng.uniform(*NOISE_SIGMA)
    noise = rng.normal(0.0, sigma, img.shape)
    return to_uint8(img.astype(np.float32) + noise)


def rot90(img):
    """n x 90 deg (never 0); optionally also h/v flip."""
    ops = ["r90", "r180", "r270"] + (["hflip", "vflip"] if USE_FLIPS else [])
    op = rng.choice(ops)
    if op == "r90":
        out = np.rot90(img, 1)
    elif op == "r180":
        out = np.rot90(img, 2)
    elif op == "r270":
        out = np.rot90(img, 3)
    elif op == "hflip":
        out = img[:, ::-1]
    else:
        out = img[::-1, :]
    return np.ascontiguousarray(out)


def chain(*fns):
    def _f(img):
        for f in fns:
            img = f(img)
        return img
    return _f


# ----------------- Augmentation list (10) -----------------
AUGMENTATIONS = [
    ("GaussianBlur",             gaussian_blur),
    ("Contrast",                 contrast),
    ("Compress",                 compress),
    ("GradientShade",            gradient_shade),
    ("WhiteNoise",               white_noise),
    ("Rot90",                    rot90),
    # ---- Combinations ----
    ("Combo_Compress_Rot90",     chain(compress, rot90)),          # -> vertical compression
    ("Combo_Shade_Noise",        chain(gradient_shade, white_noise)),
    ("Combo_Blur_Contrast_Noise", chain(gaussian_blur, contrast, white_noise)),
    ("Combo_Comp_Rot_Shade_Noise", chain(compress, rot90, gradient_shade, white_noise)),
]


# ----------------- IO / collage -----------------
def maybe_write(path: Path, img):
    if (not OVERWRITE) and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def make_collage(tiles, cols=COLLAGE_COLS, scale=COLLAGE_SCALE):
    """tiles: list of (label, image). Returns one BGR image with a labelled grid."""
    h, w = tiles[0][1].shape[:2]
    tw, th = w * scale, h * scale
    label_h, gap = 22, 4
    rows = math.ceil(len(tiles) / cols)

    canvas = np.full(
        (rows * (th + label_h + gap) + gap, cols * (tw + gap) + gap, 3),
        255, dtype=np.uint8
    )

    for k, (label, im) in enumerate(tiles):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(im, (tw, th), interpolation=cv2.INTER_NEAREST)

        r, c = divmod(k, cols)
        x0 = gap + c * (tw + gap)
        y0 = gap + r * (th + label_h + gap)
        canvas[y0 + label_h:y0 + label_h + th, x0:x0 + tw] = tile

        fs = 0.45
        while cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > tw - 4 and fs > 0.25:
            fs -= 0.02
        cv2.putText(canvas, label, (x0 + 2, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def write_readme():
    if not WRITE_README:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    readme = OUTPUT_ROOT / "README_augmentations.txt"
    lines = [
        "Augmentation pipeline\n\n",
        "Total augmentations per image: {}\n".format(len(AUGMENTATIONS)),
        "Original image saved: {}\n\n".format(SAVE_ORIGINALS),
        "Augmentations:\n",
    ]
    for i, (name, _) in enumerate(AUGMENTATIONS, 1):
        lines.append(f"  {i:02d}. {name}\n")
    readme.write_text("".join(lines), encoding="utf-8")


def main():
    print("[INFO] Input root :", DATA_ROOT)
    print("[INFO] Output root:", OUTPUT_ROOT)
    print("[INFO] Classes    :", INPUT_FOLDERS)
    print("[INFO] Augmentations per image:", len(AUGMENTATIONS))
    print(f"[INFO] Collage: {MAKE_COLLAGE} ({COLLAGE_PER_FOLDER} per folder) -> {COLLAGE_DIR}")
    print()

    write_readme()

    for folder in INPUT_FOLDERS:
        in_dir = DATA_ROOT / folder
        if not in_dir.exists():
            print(f"[WARN] Missing folder: {in_dir}")
            continue

        out_dir = OUTPUT_ROOT / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in in_dir.rglob("*") if is_image_file(p))
        print(f"[FOLDER] {folder}: {len(files)} images")

        for i, p in enumerate(files, start=1):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue

            rel  = p.relative_to(in_dir)
            base = rel.stem
            ext  = rel.suffix

            if SAVE_ORIGINALS:
                maybe_write(out_dir / rel.parent / f"{base}_orig{ext}", img)

            tiles = [("Original", img)]

            for idx, (name, aug) in enumerate(AUGMENTATIONS, 1):
                try:
                    aug_img = aug(img)
                except Exception as e:
                    print(f"[ERR] {name} failed on {p.name}: {e}")
                    continue

                out_name = f"{base}_aug{idx:02d}_{name}{ext}"
                maybe_write(out_dir / rel.parent / out_name, aug_img)
                tiles.append((f"{idx:02d} {name}", aug_img))

            # ----- Collage preview -----
            if MAKE_COLLAGE and i <= COLLAGE_PER_FOLDER:
                collage = make_collage(tiles)
                maybe_write(COLLAGE_DIR / f"{folder}_{base}_collage.png", collage)
            # ---------------------------

            if i % 200 == 0:
                print(f"  ... {i}/{len(files)} done")

    print("\n[DONE] Augmented dataset written to:")
    print(" ", OUTPUT_ROOT)


if __name__ == "__main__":
    main()