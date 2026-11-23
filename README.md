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

## Datasets

`datasets/` holds the image sets (about 360 MB), all 60 x 60 grayscale PNG:

| Folder | Images | Contents |
|---|---|---|
| `shapes_dataset_60x60/` | 4,000 | 1,000 original drawings per class (training pool) |
| `shapes_aug_dataset_60x60/` | 40,000 | ten augmentations of each original (training) |
| `shapes_test_60x60/` | 1,000 | 250 test drawings per class, separately generated |
| `shapes_aug_test_60x60/` | 10,000 | ten augmentations of each test drawing (the 10,000-image board test set) |

The train/validation/test split by drawing is in `models/cnn/dwpw_cnn_60x60_v1/split.json`. The dataset scripts are `code/cnn/generate_dataset.py` and `code/cnn/augment_data.py`.

## How to run the project

**You need:** a Windows PC with Python, Vitis and Vivado 2026.1, and a PYNQ-Z2 board connected by USB.

**Before you start:** the scripts were written with absolute paths on the author's machine (for example `D:\EmbeddedSystems_E26\...`). Change the `ROOT`, `BASE` and `SERIAL_PORT` constants at the top of each script to match your folders and COM port.

1. **Python packages.** Install `code/cnn/requirements.txt` for the CNN and `code/snn/requirements.txt` for the SNN, for example `py -m pip install -r code/snn/requirements.txt`.
2. **Train the CNN** (optional, the trained weights are in `code/cnn/`): `py code/cnn/train_cnn.py`.
3. **Train the SNN** (optional, the trained weights are in `code/snn/fpga_weights.h`): `py code/snn/snn_mlp.py`.
4. **Make the SNN test features** for the C testbench: `py code/snn/make_snn_txt.py`. Point the testbench's `root` (`code/snn/snn_mlp_tb.cpp`) at the folder it writes.
5. **Vitis HLS 2026.1, C simulation and synthesis.** Make one component per network, part `xc7z020clg400-1`, clock 12 ns:
   - CNN: sources `code/cnn/ds_cnn.cpp` and `code/cnn/ds_cnn_weights.h`, testbench `code/cnn/ds_cnn_tb.cpp`, top `ds_cnn`.
   - SNN: sources `code/snn/snn_mlp.cpp` and `code/snn/fpga_weights.h`, testbench `code/snn/snn_mlp_tb.cpp`, top `snn_mlp`.
   Run C simulation, C synthesis, then Package with the `ip_catalog` format.
6. **Vivado.** Unzip `hardware/CNN_FPGA_Shapes.zip` and `hardware/SNN_FPGA_Shapes.zip`, then open each `.xpr`. Add the packaged IP from step 5 under Settings → IP → Repository if Vivado can't find it. Generate the bitstream, then File → Export → Export Hardware with "Include bitstream". The exported `.xsa` files are already in `hardware/`.
7. **Vitis application.** Create a platform from the `.xsa`, using FSBL initialization. Create an application from `code/board_app/cnn_helloworld.c` or `code/board_app/snn_helloworld.c`, build it, and run it on the board. The startup printout shows `COUNTS_PER_SECOND`.
8. **PC test.** Close any serial terminal first, then run the matching sender:
   - `py code/tools/send_images_uart_cnn.py` (CNN, 4 Mbaud)
   - `py code/tools/send_images_uart_snn.py` (SNN, 4 Mbaud)
   Each prints the accuracy, confusion matrix and timing, and writes a per-image CSV to a `results/` folder next to the script (`code/tools/results/`). Move the CSV into the top-level `results/` folder if you want to keep it with the other results.

**Not done:** the DFX swap. The CNN and SNN are separate bitstreams, and their ports and register addresses are identical so that a DFX region could be added later.

