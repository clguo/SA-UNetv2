Here is the complete, correctly formatted `README.md` content. I have cleaned up the messy sections (Datasets, Results, Training) and formatted them into proper tables and code blocks.

-----

````markdown
# SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2509.11774-b31b1b.svg)](https://arxiv.org/abs/2509.11774)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[cite_start]This repository contains the official implementation of **SA-UNetv2** (arXiv 2025)[cite: 30, 31].

[cite_start]**SA-UNetv2** is a lightweight, efficient, and robust architecture tailored for retinal vessel segmentation[cite: 41]. [cite_start]It introduces **Cross-scale Spatial Attention (CSA)** into all skip connections and utilizes a compound **BCE + MCC loss** to handle severe class imbalance[cite: 41, 96].

![Performance Comparison](images/DRIVEP.png)
*Figure 1: Comparison of retinal vessel segmentation networks on the DRIVE dataset. [cite_start]SA-UNetv2 achieves the highest F1 score with the lowest model complexity (0.26M parameters)[cite: 72, 73].*

## 🌟 Key Features

* [cite_start]**Lightweight:** Only **0.26M parameters** and **1.2MB** memory footprint (approx. 50% reduction compared to SA-UNet)[cite: 42].
* [cite_start]**Efficient:** Sub-second inference (approx 0.95s) on CPU for 592x592 resolution[cite: 149].
* [cite_start]**Cross-scale Spatial Attention (CSA):** Bridges the semantic gap between encoder and decoder features by integrating attention across pathways[cite: 96, 105].
* [cite_start]**Improved Convolutional Block:** Adopts `Conv 3x3` → `DropBlock` → `Group Normalization` → `SiLU` for better stability and gradient flow[cite: 101].
* [cite_start]**Robust Loss:** Combines Binary Cross-Entropy (BCE) and Matthews Correlation Coefficient (MCC) loss to tackle vessel-background imbalance[cite: 116, 120].

## ARCHITECTURE

![Framework](images/SA-UNetv2.pdf)
*The architecture of SA-UNetv2. [cite_start]It integrates Cross-scale Spatial Attention (CSA) in skip connections and uses an optimized convolutional block design[cite: 94, 105].*

## 🛠️ Installation

Our environment is based on TensorFlow/Keras. Please install the dependencies as follows:

```bash
# Core requirements
pip install tensorflow==2.12.0
pip install keras_cv==0.5.0
pip install keras-flops
````

## 📂 Datasets

[cite\_start]We use the **DRIVE** and **STARE** datasets[cite: 128]. [cite\_start]Following the paper, we apply specific augmentations (zero-padding) to ensure consistent input sizes[cite: 132].

| Dataset | Description | Download Link |
| :--- | :--- | :--- |
| **Original DRIVE** | With FOV masks | [Google Drive](https://drive.google.com/file/d/1fNaGhYTQ5dUA1X38U7yhVdYNwiISdWPR/view) |
| **Augmented DRIVE** | Pre-processed (592x592) | [Google Drive](https://drive.google.com/file/d/1t_UxlVWZXBtJQQNxW0vPdwrnqcdYdrRs/view) |
| **Augmented STARE** | Pre-processed (704x704) | [Google Drive](https://drive.google.com/file/d/1-704iIpSQfuVlfKR1p8HUW4cYCDliNLr/view) |

> **Note:**
> [cite\_start]\* **DRIVE:** Images are zero-padded to **592x592**[cite: 132].
> [cite\_start]\* **STARE:** Images are zero-padded to **704x704**[cite: 132].
> [cite\_start]\* During testing, outputs are cropped back to original sizes (DRIVE: 584x565, STARE: 700x605)[cite: 133].

## 🚀 Training & Evaluation

### Training Configuration

[cite\_start]Following the paper settings[cite: 138, 140]:

  * **Optimizer:** Adam (Learning rate 1e-3)
  * **Loss Function:** $0.5 \times \mathcal{L}_{BCE} + 0.5 \times \mathcal{L}_{MCC}$
  * **Batch Size:** 8 (DRIVE), 2 (STARE)
  * **Epochs:** 150 (with early stopping)
  * **Regularization:** DropBlock (rate 0.15, block size 7)

To train the model, run:

```bash
python train.py --dataset DRIVE --batch_size 8
```

## 📊 Results

### Quantitative Comparison (DRIVE)

[cite\_start]SA-UNetv2 achieves state-of-the-art performance with significantly fewer parameters[cite: 123, 149].

| Model | F1 Score | Jaccard | Params (M) | GFLOPs |
| :--- | :---: | :---: | :---: | :---: |
| U-Net | 81.30 | 68.52 | 8.64 | 137.10 |
| Attention U-Net | 81.47 | 68.76 | 8.65 | 425.05 |
| SA-UNet | 82.44 | 70.15 | 0.54 | 26.54 |
| **SA-UNetv2 (Ours)** | **82.82** | **70.69** | **0.26** | **21.19** |

### Qualitative Visualization

*Visual comparison on DRIVE (top) and STARE (bottom) datasets. [cite\_start]SA-UNetv2 demonstrates superior capability in delineating fine-grained vascular structures compared to SA-UNet[cite: 155].*

## 📝 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{guo2025saunetv2,
  title={SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation},
  author={Guo, Changlu and Christensen, Anders Nymark and Dahl, Anders Bjorholm and Yi, Yugen and Hannemose, Morten Rieger},
  journal={arXiv preprint arXiv:2509.11774},
  year={2025}
}
```

## Acknowledgements

This work builds upon the original [SA-UNet](https://github.com/clguo/SA-UNet).

```
```
