#!/usr/bin/env python3
"""
Generate the BASE (unaugmented) shapes dataset.

  60 x 60, grayscale 8-bit PNG, black pen (0) on white paper (255).
  Classes (index order, also written to classes.txt):
      0 square   1 circle   2 triangle   3 figure8
  1000 images per class by default.

Two styles, 50 % each (they alternate: even index = defined, odd index = hand;
the style is also in the file name: <class>_<index>_defined.png / _hand.png):
  defined : exact geometric shape, straight lines, closed loop.
  hand    : simulated hand-drawn shape, wavy line, and in ~40 % of images the
            loop is left slightly open (pen gap).

Both styles have random:
  - line thickness   (1.5 - 5.0 px)
  - size             (shape extent roughly 22 - 50 px)
  - position         (anywhere that fits, i.e. offset from the centre)
  - rotation         (square 0-90 deg, circle none, triangle and figure8 0-360 deg)
  - triangle shape   (slightly different corner positions)
Only the hand style also has the wobble and the pen gap.

NO photometric augmentation: no noise, no blur, no brightness change.
Run augment_data.py on the train split later, after splitting.

Usage:  python generate_dataset.py [--per-class 1000] [--seed 123] [--out <folder>]
Needs:  numpy, Pillow
"""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

N = 60                    # image size in pixels
SS = 4                    # supersampling for smooth, antialiased lines
CLASSES = ["square", "circle", "triangle", "figure8"]

LW_RANGE = (1.5, 5.0)     # line thickness in final pixels
R_RANGE = (11.0, 25.0)    # half-size of the shape in final pixels
WOBBLE_RANGE = (0.02, 0.10)   # wobble amplitude in unit-shape coordinates
GAP_PROB = 0.4            # probability that the pen leaves a small gap
MARGIN = 2.0              # minimum white border around the pen line (pixels)


def base_points(cls, rng):
    """Closed polyline in unit coordinates, roughly within [-1, 1]."""
    pts = []
    if cls == "square":
        c = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            for k in range(25):
                t = k / 25
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    elif cls == "circle":
        for k in range(100):
            t = 2 * math.pi * k / 100
            pts.append((math.cos(t), math.sin(t)))
    elif cls == "triangle":
        v = []
        for ang in (-90, 30, 150):
            a = math.radians(ang)
            v.append((math.cos(a) + rng.uniform(-0.1, 0.1),
                      math.sin(a) + rng.uniform(-0.1, 0.1)))
        for i in range(3):
            a, b = v[i], v[(i + 1) % 3]
            for k in range(30):
                t = k / 30
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    elif cls == "figure8":
        for k in range(140):
            t = 2 * math.pi * k / 140
            pts.append((0.55 * math.sin(2 * t), math.sin(t)))
    return np.array(pts, dtype=np.float64)


def wobble(pts, amp, rng):
    """Smooth, low-frequency wobble along the pen line."""
    n = len(pts)
    t = np.arange(n) / n
    dx = np.zeros(n)
    dy = np.zeros(n)
    for q in (1, 2, 3):
        dx += np.sin(2 * math.pi * q * 1.5 * t + rng.uniform(0, 6.283)) / q
        dy += np.sin(2 * math.pi * q * 1.5 * t + rng.uniform(0, 6.283)) / q
    return pts + amp * np.stack([dx, dy], axis=1)


def make_image(cls, rng, hand):
    pts = base_points(cls, rng)
    if hand:                              # only the hand style gets a wavy line
        pts = wobble(pts, rng.uniform(*WOBBLE_RANGE), rng)

    if cls == "square":
        rot = rng.uniform(0, math.pi / 2)
    elif cls == "circle":
        rot = 0.0
    else:
        rot = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(rot), math.sin(rot)
    pts = pts @ np.array([[c, s], [-s, c]])

    lw = rng.uniform(*LW_RANGE)
    r = rng.uniform(*R_RANGE)
    while True:                           # shrink until it fits with the margin
        p = pts * r
        half = lw / 2 + MARGIN
        lo, hi = p.min(axis=0), p.max(axis=0)
        if (hi - lo).max() + 2 * half <= N:
            break
        r *= 0.92
    cx = rng.uniform(half - lo[0], N - half - hi[0])     # random offset from centre
    cy = rng.uniform(half - lo[1], N - half - hi[1])
    p = (p + np.array([cx, cy])) * SS

    if hand and rng.random() < GAP_PROB:  # pen gap where the loop should close
        drop = max(1, int(len(p) * rng.uniform(0.02, 0.06)))
        line = [tuple(x) for x in p[:-drop]]
    else:
        line = [tuple(x) for x in p] + [tuple(p[0])]

    img = Image.new("L", (N * SS, N * SS), 0)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(lw * SS)))
    d.line(line, fill=255, width=w, joint="curve")
    for end in (line[0], line[-1]):
        d.ellipse([end[0] - w / 2, end[1] - w / 2, end[0] + w / 2, end[1] + w / 2], fill=255)
    ink = np.asarray(img.resize((N, N), Image.BOX), dtype=np.float64) / 255.0
    return Image.fromarray(np.round(255 * (1 - ink)).astype(np.uint8), "L"), lw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "shapes_dataset_60x60")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    seen = set()
    for cls in CLASSES:
        (a.out / cls).mkdir(exist_ok=True)
        made = 0
        while made < a.per_class:
            hand = (made % 2 == 1)        # 50 % defined, 50 % hand-drawn
            im, _ = make_image(cls, rng, hand)
            h = hashlib.md5(im.tobytes()).hexdigest()
            if h in seen:                 # never write an exact duplicate
                continue
            seen.add(h)
            style = "hand" if hand else "defined"
            im.save(a.out / cls / f"{cls}_{made:04d}_{style}.png")
            made += 1
        print(f"{cls}: {made} images ({(made + 1) // 2} defined, {made // 2} hand)")
    print(f"Done. {a.per_class * len(CLASSES)} images, {N}x{N}, in {a.out}")


if __name__ == "__main__":
    main()
