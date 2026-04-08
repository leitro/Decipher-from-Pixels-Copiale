# Learning to Decipher from Pixels: A Case Study of Copiale

This repository contains the code and dataset metadata for the paper ["Learning to Decipher from Pixels: A Case Study of Copiale"](), accepted to International Conference on Historical Cryptology (HistoCrypt) 2026.

The project studies transcription-free decipherment of historical handwritten ciphers. Instead of first transcribing cipher symbols and then applying cryptanalysis, the model learns a direct mapping from cipher line images to plaintext using TrOCR.

## Overview

The training pipeline has two stages:

1. Pretrain on a unified handwritten text-line corpus built from four public datasets: IAM, CVL, RIMES, and EU27.
2. Fine-tune the pretrained model on line-level Copiale image-to-plaintext pairs.

The main training script is `main_trocr.py`.

## Repository Contents

- `main_trocr.py`: training, evaluation, checkpointing, and attention visualization code.
- `unified_line_iam_cvl_rimes_eu27.txt`: manifest for the handwriting pretraining corpus.
- `copiale_gt/train.gt`, `copiale_gt/valid.gt`, `copiale_gt/test.gt`: Copiale line-level plaintext ground truth splits.

## Method

We use `microsoft/trocr-base-handwritten` as the base model and train it in two stages:

- Stage I: handwriting pretraining on 66,492 text lines from IAM, CVL, RIMES, and EU27.
- Stage II: Copiale fine-tuning on line-level cipher image/plaintext pairs.

The current script configuration is:

- `EPOCH = 200`
- `PATIENCE = 20`
- `BATCH_SIZE = 24`
- optimizer: `AdamW`
- metrics: CER and WER

During training, the script saves the best checkpoint by validation CER and then evaluates on train, validation, and test sets. It also exports attention visualizations to `./visualizations/`.

## Installation

Create a Python environment and install the required packages:

```bash
pip install torch transformers pillow pandas jiwer matplotlib numpy scikit-learn tqdm
```

The code is implemented in Python using PyTorch and Hugging Face Transformers. A CUDA-enabled GPU is highly recommended for training; for example, we use an NVIDIA 4090 (24 GB) to train with a batch size of 24.


## Data Preparation

### 1. Handwriting Pretraining Data

The file `unified_line_iam_cvl_rimes_eu27.txt` contains one text line per sample in the following format:

```text
/absolute/path/to/image.jpg<TAB>transcription
```

This manifest mixes four public handwritten text-line datasets:

- IAM
- CVL
- RIMES
- EU27

Please download these datasets from their official sources and update the image paths in the manifest so they point to valid files on your system.

**Important:** the paths in this file are set to our local folder as absolute paths, please modify them to match your own directory structure before using the script.


### 2. Copiale Fine-Tuning Data

The Copiale line-level data is organized through `.gt` files in `copiale_gt/`:

- training: 1,269 lines
- validation: 175 lines
- test: 370 lines

Each entry has the format:

```text
image_id<TAB>plaintext
```

For Copiale, the script expects line images at:

```text
copiale_gt/crop_lines/<image_id>.png
```

Example:

```text
1-1    : des
```

corresponds to:

```text
copiale_gt/crop_lines/1-1.png
```

Copiale line-level dataset download: [Copiale Lines Dataset](https://huggingface.co/datasets/leitro/Copiale_Lines)

## Expected Directory Layout

Your local repository should look like this after preparing the datasets:

```text
Decipher-from-Pixels-Copiale/
├── main_trocr.py
├── unified_line_iam_cvl_rimes_eu27.txt
└── copiale_gt/
    ├── train.gt
    ├── valid.gt
    ├── test.gt
    └── crop_lines/
        ├── 1-1.png
        ├── 1-2.png
        └── ...
```

## Training

### Stage I: Pretraining on Handwritten Data

```bash
python main_trocr.py --dataset unified
```

This stage:

- reads the unified handwriting manifest
- randomly splits the data into train/validation/test with an 80/10/10 ratio
- trains TrOCR from `microsoft/trocr-base-handwritten`
- saves the best checkpoint to `trocr_unified_best.pt`

### Stage II: Fine-Tuning on Copiale

```bash
python main_trocr.py --dataset copiale
```

This stage:

- loads the pretrained checkpoint `trocr_unified_best.pt`
- fine-tunes on the Copiale line-level dataset
- saves the best checkpoint to `trocr_copiale_best.pt`

## Outputs

After training, the script produces:

- `trocr_unified_best.pt`: best checkpoint from handwriting pretraining
- `trocr_copiale_best.pt`: best checkpoint from Copiale fine-tuning
- `visualizations/`: attention visualization images for train, validation, and test samples

The script also prints:

- average training loss per epoch
- validation CER and WER per epoch
- final CER and WER on train, validation, and test splits

## Notes

- The repository does not include the actual pretraining images from IAM, CVL, RIMES, and EU27.
- The repository does not include the Copiale line images under `copiale_gt/crop_lines/`.

## Citation

If you use this repository, dataset splits, or paper, please cite:

```bibtex
@inproceedings{kang2026learning,
  title     = {Learning to Decipher from Pixels: A Case Study of Copiale},
  author    = {Lei Kang, Giuseppe De Gregorio$, Raphaela Heil, Alicia Fornés, Beáta Megyesi$},
  booktitle = {International Conference on Historical Cryptology (HistoCrypt)},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE`.

