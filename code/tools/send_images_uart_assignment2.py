#!/usr/bin/env python3
"""
PC -> UART -> PYNQ-Z2 (CPU -> BRAM -> DS-CNN IP) -> UART -> PC

Sends 60x60 grayscale test images as int16 Q8.8, receives the prediction,
accelerator time and logits, and reports accuracy, per-class accuracy,
latency, throughput and UART/communication overhead (measured separately).

Usage (Windows):
    python send_images_uart.py --port COM5
    python send_images_uart.py --port COM5 --max 20          # quick test
    python send_images_uart.py --port COM5 --folder-mode     # use every image in --root
"""

from pathlib import Path
import argparse
import csv
import json
import struct
import time

import cv2
import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

REPO_DIR = Path(r"E:\Emb_sys_A2\E26_assignment_train_cnn-main")

# Original (non-augmented) images; the test IDs come from split.json,
# so this is the SAME test set used when evaluating the trained model.
DEFAULT_ROOT = REPO_DIR / "shapes_dataset_60x60"
DEFAULT_SPLIT = REPO_DIR / "cnn_runs" / "dwpw_cnn_60x60_v2" / "split.json"

DEFAULT_PORT = "COM5"          # check Device Manager -> Ports (COM & LPT)
BAUD_RATE = 4000000            # must match UART_BAUD in helloworld.c

IMG_H = 60
IMG_W = 60
N_PIXELS = IMG_H * IMG_W       # 3600
TX_BYTES = N_PIXELS * 2        # 7200 bytes, int16 payload

FRAME_START = b"\xA6\x6A"
REPLY_START = 0xA5
REPLY_END = 0x5A

# Must match the training order (train_cnn.py / run_info.txt)
CLASS_FOLDERS = [
    "square",       # 0
    "circle",       # 1
    "figure8",      # 2
    "triangle",     # 3
]
N_CLASSES = len(CLASS_FOLDERS)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# IMAGE PREPROCESSING
# Match training:
#   x = u8 / 255
#   z = (x - 0.5) / 0.25 = 4*x - 2
# Send Q8.8 (ap_fixed<16,8>):
#   q = round(z * 256) = round((1024*u8)/255 - 512)
# ============================================================

def image_to_q8_8(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    px = img.astype(np.int32).reshape(-1)
    q = ((1024 * px + 127) // 255 - 512)
    q = np.clip(q, -32768, 32767).astype(np.int16)

    assert q.shape[0] == N_PIXELS
    return q


# ============================================================
# UART COMMUNICATION
# PC -> FPGA:
#   0xA6 0x6A <len u16 LE> <3600 x int16 LE>          (7204 bytes)
# FPGA -> PC:
#   0xA5 <pred u8> <time_us u32 LE> <4 x logit i16 LE> 0x5A   (15 bytes)
# ============================================================

def send_image_and_get_prediction(ser, q_img):
    payload = q_img.astype("<i2", copy=False).tobytes(order="C")

    if len(payload) != TX_BYTES:
        raise RuntimeError(f"Bad payload size: {len(payload)} != {TX_BYTES}")

    packet = FRAME_START + struct.pack("<H", len(payload)) + payload

    ser.reset_input_buffer()

    t_start = time.perf_counter()
    ser.write(packet)
    ser.flush()
    t_sent = time.perf_counter()

    # Wait for the binary reply start byte; skips the board's ASCII debug
    # print ("accel: ... ms ..."), which never contains 0xA5.
    deadline = time.time() + 5.0
    found_start = False

    while time.time() < deadline:
        b = ser.read(1)
        if b and b[0] == REPLY_START:
            found_start = True
            break

    if not found_start:
        raise RuntimeError("Timeout waiting for FPGA reply start 0xA5 "
                           "(wrong COM port? program not running? baud mismatch?)")

    rest = ser.read(1 + 4 + 8 + 1)
    t_done = time.perf_counter()

    if len(rest) != 14:
        raise RuntimeError("Incomplete FPGA reply")

    if rest[-1] != REPLY_END:
        raise RuntimeError(f"Bad FPGA reply end: 0x{rest[-1]:02X}")

    pred = rest[0]
    time_us = struct.unpack("<I", rest[1:5])[0]
    logits = struct.unpack("<hhhh", rest[5:13])

    timing = {
        "tx_ms": (t_sent - t_start) * 1000.0,          # PC -> board transfer
        "roundtrip_ms": (t_done - t_start) * 1000.0,    # whole request
    }
    return pred, time_us, logits, timing


# ============================================================
# TEST SET
# ============================================================

def load_test_set(root: Path, split_path: Path, folder_mode: bool):
    items = []

    if not folder_mode:
        split = json.loads(split_path.read_text())
        for img_id in split["test"]:
            cname = img_id.split("_")[0]
            items.append((root / cname / f"{img_id}.png", CLASS_FOLDERS.index(cname)))
        return items

    for label, cname in enumerate(CLASS_FOLDERS):
        class_dir = root / cname
        if not class_dir.exists():
            print(f"Skipping missing folder: {class_dir}")
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        items.extend((p, label) for p in files)
    return items


def limit_per_class(items, max_per_class):
    if max_per_class is None:
        return items
    out, counts = [], [0] * N_CLASSES
    for path, label in items:
        if counts[label] < max_per_class:
            out.append((path, label))
            counts[label] += 1
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Send test images to the PYNQ-Z2 CNN over UART")
    ap.add_argument("--port", default=DEFAULT_PORT, help="serial port, e.g. COM5")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="image root folder")
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="split.json with test IDs")
    ap.add_argument("--folder-mode", action="store_true",
                    help="ignore split.json and send every image in --root/<class>/")
    ap.add_argument("--max", type=int, default=None, help="max images per class")
    ap.add_argument("--csv", type=Path, default=Path("fpga_cnn_results.csv"),
                    help="per-image results file")
    args = ap.parse_args()

    items = limit_per_class(load_test_set(args.root, args.split, args.folder_mode), args.max)

    wire_bits = (len(FRAME_START) + 2 + TX_BYTES) * 10      # 8N1 = 10 bits/byte
    theory_tx_ms = wire_bits / BAUD_RATE * 1000.0

    print()
    print("=====================================")
    print(" FPGA UART CNN TEST")
    print(f" Q8.8 int16 {IMG_H}x{IMG_W} image sender")
    print("=====================================")
    print(f"Image root   : {args.root}")
    print(f"Test set     : {'all files (folder mode)' if args.folder_mode else args.split}")
    print(f"Images       : {len(items)}")
    print(f"Serial port  : {args.port} @ {BAUD_RATE} baud (8N1)")
    print(f"TX bytes/img : {TX_BYTES} payload + 4 header")
    print(f"Theoretical TX time: {theory_tx_ms:.2f} ms/img")
    print()

    if not items:
        print("No images found - check --root / --split.")
        return

    ser = serial.Serial(
        port=args.port,
        baudrate=BAUD_RATE,
        timeout=5,
        write_timeout=5,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    accel_us, tx_ms, rt_ms = [], [], []
    rows = []

    t_run_start = time.perf_counter()
    try:
        for path, true_label in items:
            q_img = image_to_q8_8(path)
            pred, time_us, logits, timing = send_image_and_get_prediction(ser, q_img)

            if 0 <= pred < N_CLASSES:
                confusion[true_label, pred] += 1
            ok = pred == true_label

            accel_us.append(time_us)
            tx_ms.append(timing["tx_ms"])
            rt_ms.append(timing["roundtrip_ms"])
            rows.append([path.name, true_label, pred, int(ok), time_us,
                         f"{timing['tx_ms']:.3f}", f"{timing['roundtrip_ms']:.3f}", *logits])

            pred_name = CLASS_FOLDERS[pred] if pred < N_CLASSES else str(pred)
            print(f"{path.name}: true={CLASS_FOLDERS[true_label]:8s} "
                  f"pred={pred_name:8s} "
                  f"accel={time_us} us  roundtrip={timing['roundtrip_ms']:.2f} ms  "
                  f"logits={logits}  {'OK' if ok else 'WRONG'}")
    finally:
        ser.close()
    t_run = time.perf_counter() - t_run_start

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred", "correct", "accel_us", "tx_ms",
                    "roundtrip_ms", "logit0", "logit1", "logit2", "logit3"])
        w.writerows(rows)

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accel_ms = np.array(accel_us) / 1000.0
    rt = np.array(rt_ms)

    print()
    print("=====================================")
    print(" FINAL RESULTS")
    print("=====================================")
    print(f"Total   : {total}")
    print(f"Correct : {correct}")
    if total:
        print(f"Accuracy: {correct / total * 100:.2f}%")

    print("\nPer-class accuracy")
    for i, cname in enumerate(CLASS_FOLDERS):
        n = confusion[i].sum()
        if n:
            print(f"  {cname:8s}: {confusion[i, i] / n * 100:6.2f}%  ({confusion[i, i]}/{n})")

    print("\nConfusion matrix (rows=true, cols=pred)")
    print("          " + " ".join(f"{c[:6]:>6s}" for c in CLASS_FOLDERS))
    for i, cname in enumerate(CLASS_FOLDERS):
        print(f"  {cname[:8]:8s}" + " ".join(f"{v:6d}" for v in confusion[i]))

    print("\nTiming (mean / min / max)")
    print(f"  Inference (FPGA IP)   : {accel_ms.mean():8.3f} / {accel_ms.min():.3f} / {accel_ms.max():.3f} ms")
    print(f"  UART TX (PC write)    : {np.mean(tx_ms):8.3f} ms   (theory {theory_tx_ms:.2f} ms)")
    print(f"  Round trip (total)    : {rt.mean():8.3f} / {rt.min():.3f} / {rt.max():.3f} ms")
    print(f"  Comm + CPU overhead   : {(rt - accel_ms).mean():8.3f} ms  (round trip - inference)")
    print("\nThroughput")
    print(f"  Accelerator only      : {1000.0 / accel_ms.mean():10.1f} img/s")
    print(f"  End-to-end via UART   : {total / t_run:10.1f} img/s")
    print(f"\nPer-image results saved to: {args.csv.resolve()}")
    print()


if __name__ == "__main__":
    main()
