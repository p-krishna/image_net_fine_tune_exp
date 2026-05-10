# GTSRB Traffic Sign Classifier — Training Guide

## Environment

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| torchvision | 0.15+ |
| Pillow | 9.0+ |
| pandas | 1.5+ |
| matplotlib | 3.6+ |

Install all dependencies:
```bash
pip install torch torchvision pillow pandas matplotlib
```

---

## Project Folder Structure

```
gtsrb/
├── config.py                        # All hyperparameters and paths
├── dataset.py                       # DataLoader and label remapping
├── model.py                         # AlexNet / VGG16 builder
├── train.py                         # Training loop
├── evaluate.py                      # Accuracy and confusion matrix
├── sampler.py                       # Sample / reset dataset
├── main.py                          # Entry point — runs all experiments
├── visualize.py                     # Class distribution plot
├── visualize_gradcam_before_after.py  # Grad-CAM visualizations
├── visualize_preprocessing.py       # Preprocessing step images
├── data/
│   ├── Train/                       # Subfolders 0–42
│   ├── Test/
│   ├── Train.csv
│   └── Test.csv
├── checkpoints/
│   ├── alexnet_exp1/epoch_5.pt
│   ├── alexnet_exp2/epoch_5.pt
│   ├── vgg16_exp1/epoch_5.pt
│   └── vgg16_exp2/epoch_5.pt
└── results/
    ├── class_distribution.png
    ├── confusion_matrix_alexnet_exp1.png
    └── ...
```

---

## Class Label Mapping

The 43 original GTSRB classes are collapsed into 3 shape categories.
6 labels are excluded entirely (6, 12, 14, 32, 41, 42).

| New Class | Label ID | Original GTSRB Labels |
|---|---|---|
| Circles | 0 | 0–5, 7–10, 15, 16, 17 |
| Triangles | 1 | 11, 13, 18–31 |
| Blue Signs | 2 | 33–40 |
| *(Excluded)* | — | 6, 12, 14, 32, 41, 42 |

---

## Experiment Hyperparameters

| Experiment | Model | Learning Rate | Batch Size | Epochs | Augmentation |
|---|---|---|---|---|---|
| exp1 | AlexNet | 1e-3 | 32 | 5 | None |
| exp2 | AlexNet | 1e-5 | 64 | 5 | Flip + Translate |
| exp3 | VGG16 | 1e-3 | 32 | 5 | None |
| exp4 | VGG16 | 1e-5 | 64 | 5 | Flip + Translate |

All experiments use a **frozen backbone** — only the final `Linear(→3)` head is trained.

---

## Step-by-Step: Sample Run (Recommended First)

### 1. Generate a sample dataset
Creates a stratified ~10% subset for fast testing (~10 seconds).
```bash
python sampler.py --sample
```

### 2. Train on sample
Runs all 4 experiments on the small dataset to verify the pipeline works.
```bash
python main.py
```
Expected time: **5–10 minutes** per experiment on CPU.

### 3. Check outputs
```bash
ls results/
ls checkpoints/
```
You should see `.png` plots and `.pt` checkpoint files.

---

## Step-by-Step: Full Data Training

### 1. Reset to full dataset
```bash
python sampler.py --reset
```

### 2. Train with reduced CPU priority (prevents overheating)
```bash
nice -n 15 python main.py
```
Expected time: **30–60 minutes** per experiment on CPU.

> **Tip — reduce CPU load:**  
> Add these two lines at the top of `main()` in `main.py` to limit thread usage:
> ```python
> import torch
> torch.set_num_threads(4)
> torch.set_num_interop_threads(2)
> ```

---

## Running a Single Experiment

To re-run only one experiment without starting from scratch, comment out
the others in `main.py` or call the training function directly:

```bash
python -c "
from main import run_experiment
from config import EXPERIMENTS
run_experiment(EXPERIMENTS[1])  # 0=exp1, 1=exp2, 2=exp3, 3=exp4
"
```

---

## Grad-CAM Visualization

Generates before/after heatmaps showing what the model attends to.

```bash
# AlexNet
python visualize_gradcam_before_after.py \
    --image data/Train/0/00000_00000_00029.png \
    --model alexnet

# VGG16
python visualize_gradcam_before_after.py \
    --image data/Train/0/00000_00000_00029.png \
    --model vgg16
```

Output saved to `results/gradcam_before_after_<model>_<image>/`:

| File | Description |
|---|---|
| `00_original.png` | Resized input image |
| `01_before_heatmap.png` | Grad-CAM map — random head (before training) |
| `02_before_overlay.png` | Heatmap blended on image — before training |
| `03_after_heatmap.png` | Grad-CAM map — fine-tuned head (after exp1) |
| `04_after_overlay.png` | Heatmap blended on image — after exp1 |
| `05_summary_side_by_side.png` | 2×3 summary grid — use this in your report |

---

## Preprocessing Visualization

Saves each preprocessing step as a separate image file.
Automatically picks the largest image in the folder you specify.

```bash
python visualize_preprocessing.py --folder data/Train/0
python visualize_preprocessing.py --folder data/Train/18
python visualize_preprocessing.py --folder data/Train/33
```

Output steps saved to `results/preprocessing_<class>/`:
`step_01_original.png` → `step_07_normalize.png`

---

## All Outputs

| File | Description |
|---|---|
| `results/class_distribution.png` | Train/test class balance bar chart |
| `results/confusion_matrix_<model>_exp<N>.png` | Confusion matrix |
| `results/gradcam_before_after_*/` | Grad-CAM before/after images |
| `results/preprocessing_*/` | Per-step preprocessing images |
| `checkpoints/<model>_exp<N>/epoch_5.pt` | Saved model weights |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `IndexError: list index out of range` in Grad-CAM | Backward hook didn't fire — frozen backbone blocks gradients | Temporarily unfreeze last conv layer before `.backward()` |
| `RuntimeError: Output 0 of BackwardHookFunctionBackward is a view...` | AlexNet/VGG16 use `ReLU(inplace=True)` which conflicts with backward hooks | Call `_disable_inplace_relu(model)` before running Grad-CAM |
| `FileNotFoundError: checkpoints/alexnet_exp1` | Experiment hasn't been run yet | Run `python main.py` first |
| CPU at 100% / overheating | Default PyTorch uses all threads | Set `torch.set_num_threads(4)` and use `nice -n 15` |
| Training very slow (~50 min/epoch for VGG16) | Images resized on-the-fly each epoch | Run `python preprocess.py` to pre-resize and cache all images |
| `RuntimeError: CUDA not available` | No GPU — expected for this setup | Training runs on CPU — this is normal |
