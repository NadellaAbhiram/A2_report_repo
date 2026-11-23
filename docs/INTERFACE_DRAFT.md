# Assignment 2 - interface agreement (DRAFT, not final)

Status: proposal by Abhiram. Nothing here is agreed until all four people say yes.
Deadline: 2026-11-23 23:59. Board: PYNQ-Z2 (xc7z020clg400-1).
Each item has a PROPOSAL and a box to tick once agreed.

## 0. Roles (relay, but everyone starts today with dummy inputs)

| # | Person | Owns | Hands over | Can start today with |
|---|--------|------|------------|----------------------|
| 1 | ____ | Dataset (generated + hand-drawn + paper photos), labels, splits, shared preprocessing function; then CNN + SNN training | dataset, split files, trained weights, training code, golden outputs | - |
| 2 | ____ | HLS IPs for CNN and SNN, C-sim vs golden outputs, export IP | exported IP zips, synthesis reports | random weights, 20 dummy images |
| 3 | ____ | Vivado: two separate designs (CNN, SNN), bitstreams, XSA files, address map | XSA files, address map | the HLS IP with random weights |
| 4 | ____ | Vitis bare-metal C app, UART protocol, PC Python script, evaluation | app, PC script, results tables | UART echo test, dummy IP |

Rule: nobody waits for the previous person's final output. Use dummy data until it arrives.
Everyone writes their own section of the report (half a page each) and helps with the video.

## 1. Classes  [ ]
PROPOSAL (index order is fixed and used everywhere):

| idx | name |
|-----|------|
| 0 | square |
| 1 | rectangle |
| 2 | circle |
| 3 | triangle |
| 4 | figure8 |

Optional later: ellipse (idx 5). Add only if all scripts support N classes from one config.
Names are lowercase, no spaces. One shared file `classes.txt` is the single source of truth.

## 2. Image format  [ ]
PROPOSAL:
- 60 x 80 (rows x cols), grayscale, 8-bit, 0..255. Same as both labs, so least code change.
- Black shape on white background (matches pen on white paper). Same polarity in training,
  generated images and the board; padding uses white.
- Do not stretch photos to 60x80: crop a square around the shape, then pad to 4:3.
  Stretching turns circles into ellipses and squares into rectangles.
- Shared function `preprocess(image) -> 60x80 uint8` used by the dataset script,
  the PC script and the paper-photo path. One owner (person 1).
- Normalization inside the model: decide once (CNN lab uses (x/255 - 0.5)/0.25).
  The HLS code and the training code must use the same one.

## 3. Dataset  [ ]
PROPOSAL:
- Variations: size, position, rotation, line thickness, wobbly "hand-drawn" style, noise.
- Real hand-drawn: each person draws 30+ shapes (pen on paper, photo), mixed into train/val,
  and a separate set of real drawings kept ONLY in test.
- Folder layout: `dataset/{train,val,test}/<class>/xxxxx.png`
- Fixed split files and fixed random seed. Same split for CNN and SNN. No image in two splits.
- Target size: about 600 train / 100 val / 100 test per class (tune later).

## 4. Model architecture (frozen before HLS starts)  [ ]
PROPOSAL (person 1 and 2 decide together, write the final numbers here):
- CNN: start from the CNN lab network (Conv3x3 + ReLU + MaxPool, depthwise 3x3, pointwise 1x1,
  average pool, 1x1 classifier). Only change classes. Layer sizes: ______
- SNN: start from the SNN lab (Sobel ON/OFF, 5x5 cell averages -> 384 features,
  fully connected LIF layers). Time steps: ____  Decay (beta): ____  Threshold: ____  Reset: ____
  Hidden size: ____
- Output: scores per class (CNN logits, SNN membrane/spike counts). Class = argmax.
- Allowed layers: only what HLS can build (no exotic layers).

## 5. Numbers and weight files  [ ]
PROPOSAL:
- Train in float. HLS in fixed point (INT16 weights as in the SNN lab, ap_fixed for activations).
  Exact types: ______ . Rounding: ______ .
- Batch norm folded into conv weights before export (CNN).
- One C header per network: `cnn_weights.h`, `snn_weights.h`.
  Array names, shapes and memory order written in a table here (row-major, channel order: ____).
- Golden reference: person 1 saves expected outputs (scores + class) for the whole test set
  as `golden_cnn.csv`, `golden_snn.csv`. HLS C-sim and board runs compare to it.
  Tolerance: class must match; score difference within ____ .

## 6. Hardware interface  [ ]
PROPOSAL:
- Top functions: `cnn_top`, `snn_top` (same argument style).
- Both IPs: AXI-Lite control + AXI master to read the input buffer and write the result.
- Input buffer in BRAM through AXI BRAM controller (as in the SNN lab diagram). Base address: ______
  Result location: ______ . Same layout for both IPs so one Vitis app drives both.
- Clock: ____ MHz. Part: xc7z020clg400-1. Same board file.
- First: two separate designs (allowed by the assignment). Later: both in one design, then DFX bonus.

## 7. UART protocol  [ ]
PROPOSAL (person 4 owns, everyone reviews):
- Baud ______ (SNN lab: 4,000,000), 8N1, no flow control.
- Frame: `0xAA 0x55 | cmd (1B) | length (2B) | payload | checksum (1B)`
- Commands: select network (0=CNN, 1=SNN), send image, run, get result, ping.
- Reply: status, predicted class, scores, inference time (cycles), UART time.
- UART transfer is tested alone first (echo test) before the IP is connected.

## 8. Measurement  [ ]
PROPOSAL:
- Inference time: from IP start to done (CPU timer), and GPIO pulse if we have a scope.
- UART time reported separately.
- Resources and power from Vivado reports (state which report).
- Same test images, same conditions for CNN and SNN.
- Per-class accuracy and confusion matrix from the PC script, saved as CSV.

## 9. Git and files  [ ]
PROPOSAL: two repos (assignment asks for hardware/platform and application).
Folder names: `dataset/`, `training/`, `hls/`, `vivado/`, `vitis_app/`, `pc_script/`, `docs/`.
Commit small, with the owner's name in the message. Do not commit big datasets (share by drive).

## 10. Schedule (working back from 2026-11-23 23:59)  [ ]
TO DISCUSS: no fixed dates yet. We agree together when each stage (dataset, weights,
bitstreams, demo, video, report) is finished, leaving buffer time before the deadline for
corrections to the report and video.
Stretch (after baseline): faster implementation, both IPs in one design, DFX.

## 11. Open questions
- ____
