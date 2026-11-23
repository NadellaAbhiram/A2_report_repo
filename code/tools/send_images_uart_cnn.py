#!/usr/bin/env python3

from pathlib import Path
import csv
import struct
import time

import cv2
import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("D:/EmbeddedSystems_E26/cnn_repo/shapes_aug_test_60x60")

SERIAL_PORT = "COM7"
BAUD_RATE = 4000000

MAX_IMAGES_PER_FOLDER = 2500   # all held-out augmented images per class (same set as the SNN run)

IMG_H = 60
IMG_W = 60
N_PIXELS = IMG_H * IMG_W
RX_BYTES = N_PIXELS * 2   # int16 payload

FRAME_START = b"\xA6\x6A"

# Per-image results (written by main); copy this folder into the new repo
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "cnn_results.csv"

CLASS_FOLDERS = [
    "square",       # 0
    "circle",       # 1
    "figure8",      # 2
    "triangle",     # 3
]


# ============================================================
# IMAGE PREPROCESSING
# Match training:
#   x = u8 / 255
#   z = (x - 0.5) / 0.25 = 4*x - 2
# Send Q8.8:
#   q = round(z * 256)
#     = round((1024*u8)/255 - 512)
# ============================================================

def image_to_q8_8_60x60(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    img_u8 = img.astype(np.uint8).reshape(-1)

    px = img_u8.astype(np.int32)
    q = ((1024 * px + 127) // 255 - 512).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)

    assert q.shape[0] == N_PIXELS

    return q


# ============================================================
# UART COMMUNICATION
# FPGA expects:
#   0xA6 0x6A <len u16 LE> <3600 int16 little-endian>
#
# FPGA replies:
#   0xA5 <pred> <time_us u32 LE> 0x5A
# ============================================================

def send_image_and_get_prediction(ser, q_img):
    payload = q_img.astype("<i2", copy=False).tobytes(order="C")

    if len(payload) != RX_BYTES:
        raise RuntimeError(f"Bad payload size: {len(payload)} != {RX_BYTES}")

    packet = FRAME_START + struct.pack("<H", len(payload)) + payload

    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(packet)
    ser.flush()
    t_sent = time.perf_counter()

    # Wait for binary reply start 0xA5.
    # This skips any FPGA ASCII/debug prints such as:
    #   accel(sw): 1.234 ms
    deadline = time.time() + 5.0
    found_start = False

    while time.time() < deadline:
        b = ser.read(1)

        if not b:
            continue

        if b[0] == 0xA5:
            found_start = True
            break

    if not found_start:
        raise RuntimeError("Timeout waiting for FPGA reply start 0xA5")

    pred_byte = ser.read(1)
    time_bytes = ser.read(4)
    logit_bytes = ser.read(8)
    end = ser.read(1)
    t_done = time.perf_counter()

    if (
        len(pred_byte) != 1
        or len(time_bytes) != 4
        or len(logit_bytes) != 8
        or len(end) != 1
    ):        
        raise RuntimeError("Incomplete FPGA reply")

    if end[0] != 0x5A:
        raise RuntimeError(f"Bad FPGA reply end: {end}")

    pred = pred_byte[0]
    time_us = struct.unpack("<I", time_bytes)[0]
    logits = struct.unpack("<hhhh", logit_bytes)

    timing = {
        "tx_ms": (t_sent - t_start) * 1000.0,         # PC write call (buffer time, not wire time)
        "roundtrip_ms": (t_done - t_start) * 1000.0,  # write + board + reply
    }

    return pred, time_us, logits, timing


# ============================================================
# LABELS
# ============================================================

def label_from_folder(folder_name: str) -> int:
    return CLASS_FOLDERS.index(folder_name)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=====================================")
    print(" FPGA UART CNN TEST")
    print(" Q8.8 int16 60x60 image sender")
    print("=====================================")
    print()

    print(f"Dataset root : {ROOT}")
    print(f"Serial port  : {SERIAL_PORT}")
    print(f"Baud rate    : {BAUD_RATE}")
    print(f"Images/folder: {MAX_IMAGES_PER_FOLDER}")
    print(f"Image size   : {IMG_H} x {IMG_W}")
    print(f"TX bytes/img : {RX_BYTES}")
    print()

    ser = serial.Serial(
        port=SERIAL_PORT,
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

    total = 0
    correct = 0
    confusion = np.zeros((4, 4), dtype=int)
    accel_us, tx_ms, rt_ms = [], [], []
    rows = []
    t_run_start = time.perf_counter()

    try:
        for class_name in CLASS_FOLDERS:
            class_dir = ROOT / class_name

            if not class_dir.exists():
                print(f"Skipping missing folder: {class_dir}")
                continue

            true_label = label_from_folder(class_name)

            image_files = sorted(
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.png")) +
                list(class_dir.glob("*.bmp"))
            )[:MAX_IMAGES_PER_FOLDER]

            print()
            print(f"Processing class: {class_name}")
            print(f"Images selected: {len(image_files)}")
            print()

            for image_path in image_files:
                q_img = image_to_q8_8_60x60(image_path)

                pred, time_us, logits, timing = send_image_and_get_prediction(ser, q_img)
                time_ms = time_us / 1000.0

                total += 1
                ok = int(pred == true_label)
                correct += ok

                if 0 <= pred < 4:
                    confusion[true_label, pred] += 1

                accel_us.append(time_us)
                tx_ms.append(timing["tx_ms"])
                rt_ms.append(timing["roundtrip_ms"])
                rows.append([image_path.name, true_label, pred, ok, time_us,
                             f"{timing['tx_ms']:.3f}", f"{timing['roundtrip_ms']:.3f}", *logits])

                print(
                    f"{image_path.name}: "
                    f"true={CLASS_FOLDERS[true_label]} "
                    f"pred={CLASS_FOLDERS[pred] if pred < 4 else pred} "
                    f"accel={time_us} us "
                    f"roundtrip={timing['roundtrip_ms']:.2f} ms "
                    f"logits={logits} "
                    f"{'OK' if ok else 'WRONG'}"
                )

    finally:
        ser.close()

    t_run = time.perf_counter() - t_run_start

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred", "correct", "accel_us", "tx_ms",
                    "roundtrip_ms", "logit0", "logit1", "logit2", "logit3"])
        w.writerows(rows)

    accel_ms = np.array(accel_us, dtype=float) / 1000.0
    rt = np.array(rt_ms, dtype=float)
    theory_tx_ms = (4 + RX_BYTES) * 10 / BAUD_RATE * 1000.0   # frame header + payload, 8N1
    reply_ms = 15 * 10 / BAUD_RATE * 1000.0                    # 15-byte reply

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

    if total:
        print("\nTiming (mean / min / max)")
        print(f"  Inference (FPGA IP)   : {accel_ms.mean():8.3f} / {accel_ms.min():.3f} / {accel_ms.max():.3f} ms")
        print(f"  UART TX (wire, theory): {theory_tx_ms:8.3f} ms   (reply {reply_ms:.3f} ms)")
        print(f"  UART TX (PC write)    : {np.mean(tx_ms):8.3f} ms   (buffer time, not wire time)")
        print(f"  Round trip (total)    : {rt.mean():8.3f} / {rt.min():.3f} / {rt.max():.3f} ms")
        print(f"  Comm + CPU overhead   : {(rt - accel_ms).mean():8.3f} ms  (round trip - inference)")
        print("\nThroughput")
        print(f"  Accelerator only      : {1000.0 / accel_ms.mean():10.1f} img/s")
        print(f"  End-to-end (incl. PC preprocessing): {total / t_run:8.1f} img/s")
    print(f"\nPer-image results saved to: {RESULTS_CSV}")
    print()


if __name__ == "__main__":
    main()