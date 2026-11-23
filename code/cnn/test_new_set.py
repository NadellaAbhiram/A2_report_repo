#!/usr/bin/env python3
"""
Evaluate the trained CNN (v2) on a NEW folder of shape images.

Folder layout expected:  <folder>/<class>/<anything>.png   (60x60 grayscale)
Labels come from the FOLDER NAME and the training order below
(not from classes.txt, which uses a different order).

Usage (from code/):
  ../cnn_env/bin/python "(C) test_new_set.py" ../shapes_test_new_60x60
It also checks that no new image is an exact copy of an image in the
original dataset (so the test is truly unseen).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync")
RUN = BASE / "cnn_runs/dwpw_cnn_60x60_v2"
OLD_DATA = BASE / "shapes_dataset_60x60"
CLASS_NAMES = ["square", "circle", "figure8", "triangle"]   # = training order
SIZE = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()

    # --- load new images
    paths, labels = [], []
    for lab, name in enumerate(CLASS_NAMES):
        files = sorted((args.folder / name).glob("*.png"))
        if not files:
            sys.exit(f"No images in {args.folder / name}")
        paths += files
        labels += [lab] * len(files)

    X = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (SIZE, SIZE):
            sys.exit(f"Bad image: {p}")
        X.append(img)
    X = np.asarray(X)
    y = np.asarray(labels)
    print(f"Loaded {len(y)} images: {np.bincount(y)} per class {CLASS_NAMES}")

    # --- exact-duplicate check against the original dataset
    old = {hashlib.md5(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).tobytes()).hexdigest()
           for p in OLD_DATA.rglob("*.png")}
    dup = sum(hashlib.md5(a.tobytes()).hexdigest() in old for a in X)
    print(f"Exact copies of original-dataset images: {dup} (should be 0)")

    # --- predict (same normalization as training)
    xin = ((X.astype(np.float32) / 255.0 - 0.5) / 0.25)[..., None]
    model = tf.keras.models.load_model(RUN / "models/dwpw_cnn.keras")
    logits = model.predict(xin, verbose=0)
    pred = logits.argmax(1)

    print(f"\nAccuracy: {(pred == y).mean():.4f}\n")
    print(classification_report(y, pred, target_names=CLASS_NAMES, digits=3))
    print("Confusion matrix (rows = true, cols = predicted), order:", CLASS_NAMES)
    print(confusion_matrix(y, pred, labels=range(4)))

    out = args.folder / "v2_results"
    out.mkdir(exist_ok=True)
    np.save(out / "logits_keras.npy", logits)
    np.save(out / "labels.npy", y)
    (out / "files.txt").write_text("\n".join(f"{p.name},{l}" for p, l in zip(paths, labels)))
    print(f"\nSaved logits/labels/file list to {out}")


if __name__ == "__main__":
    main()
