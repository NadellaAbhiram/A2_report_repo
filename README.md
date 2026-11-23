# Embedded Systems E26 - Assignment 2: CNN vs SNN shape classifier on PYNQ-Z2

Classifies four shapes (square, circle, figure8, triangle; 60x60 grayscale) with a CNN and an SNN, both as Vitis HLS IPs on the PYNQ-Z2. The PC sends images over UART, a bare-metal app drives the IP, and the result goes back to the PC.

## Layout

| Folder | Contents |
|---|---|
| `code/cnn/` | CNN training (`train_cnn.py`), dataset tools, HLS source `ds_cnn.cpp`, testbench, weights headers |
| `code/snn/` | SNN training (`snn_mlp.py`), feature extraction, HLS source `snn_mlp.cpp`, testbench, weights |
| `code/board_app/` | Bare-metal `helloworld.c` for CNN and SNN (Vitis) |
| `code/tools/` | PC-side UART senders |
| `models/` | Trained runs (CNN v1/v2, SNN v1): models, plots, confusion matrices, reports, exported weights |
| `hardware/` | Exported `.xsa` files and the Vivado projects |
| `results/` | FPGA test results |
| `report/` | Report template, figures (state machines), drafts |
| `docs/` | Assignment brief, interface notes |

## Not included

The image datasets (about 350 MB) are left out. Regenerate them with `code/cnn/generate_dataset.py` and `code/cnn/augment_data.py`, or take them from the team repo.

Source of the files: https://github.com/CSham31/E26_assignment_train_cnn
