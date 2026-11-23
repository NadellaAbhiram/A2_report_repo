#!/usr/bin/env python3
"""
Create a train / val / test split BY ORIGINAL DRAWING (not by augmented image).

Why: each original has ~10 augmented copies. Splitting augmented images
randomly leaks the same drawing into train and validation.

Output: split.json with lists of original IDs (file stems), per split.
  - train/val: use the augmented copies of these IDs (shapes_aug_dataset_60x60)
  - test     : use the clean originals          (shapes_dataset_60x60)

Reads images? No. It only lists filenames. Writes only split.json.
"""
import json
import random
from pathlib import Path

BASE = Path("/Users/chamath/Documents/WORK/tmp.nosync/Assignment2.nosync")
ORIG_DIR = BASE / "shapes_dataset_60x60"
OUT_JSON = BASE / "cnn_runs" / "dwpw_cnn_60x60" / "split.json"

CLASSES = ["circle", "square", "triangle", "figure8"]   # label 0..3
FRACTIONS = (0.70, 0.15, 0.15)                          # train, val, test
SEED = 42


def split_ids(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = round(n * FRACTIONS[0])
    n_va = round(n * FRACTIONS[1])
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def main():
    rng = random.Random(SEED)
    split = {"seed": SEED, "classes": CLASSES,
             "train": [], "val": [], "test": []}

    for cname in CLASSES:
        stems = [p.stem for p in (ORIG_DIR / cname).glob("*.png")]
        # split "defined" and "hand" drawings separately so every split
        # gets the same mix
        for kind in ("defined", "hand"):
            group = [s for s in stems if s.endswith("_" + kind)]
            tr, va, te = split_ids(group, rng)
            split["train"] += tr
            split["val"] += va
            split["test"] += te
            print(f"{cname:9s} {kind:8s} total={len(group):4d} "
                  f"train={len(tr)} val={len(va)} test={len(te)}")

    # safety: no ID may appear in two splits
    s_tr, s_va, s_te = map(set, (split["train"], split["val"], split["test"]))
    assert not (s_tr & s_va or s_tr & s_te or s_va & s_te), "overlap!"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(split, indent=1))
    print(f"\nSaved {OUT_JSON}")
    print(f"train={len(s_tr)} val={len(s_va)} test={len(s_te)} originals")


if __name__ == "__main__":
    main()
