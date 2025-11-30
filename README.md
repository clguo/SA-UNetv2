# SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation

📄 **Paper:** https://arxiv.org/abs/2509.11774  
📜 **License:** https://www.google.com/search?q=LICENSE

This repository contains the official implementation of **SA-UNetv2** (arXiv 2025).

**SA-UNetv2** is a lightweight, efficient, and robust architecture tailored for retinal vessel segmentation.  
It introduces **Cross-scale Spatial Attention (CSA)** into all skip connections and utilizes a compound  
**BCE + MCC loss** to handle severe class imbalance.

*Figure 1: Comparison of retinal vessel segmentation networks on the DRIVE dataset.  
SA-UNetv2 achieves the highest F1 score with the lowest model complexity (0.26M parameters).*

---

## 🌟 Key Features

- **Lightweight:** Only **0.26M parameters** and **1.2MB** memory footprint  
  (≈50% reduction compared to SA-UNet)
- **Efficient:** Sub-second inference (≈0.95s) on CPU for **592×592** resolution
- **Cross-scale Spatial Attention (CSA):** Bridges the semantic gap between encoder and decoder
- **Improved Convolutional Block:**  
  `Conv 3×3 → DropBlock → GroupNorm → SiLU`
- **Robust Loss:** **BCE + MCC loss** to tackle severe vessel–background imbalance

---

## 🏗️ Architecture

*The architecture of SA-UNetv2 integrating CSA into skip connections with an optimized convolutional block design.*

---

## 🛠️ Installation

Environment based on **TensorFlow / Keras**:

```bash
pip install tensorflow==2.12.0
pip install keras_cv==0.5.0
pip install keras-flops


## 📂 Datasets

We use the **DRIVE** and **STARE** datasets. All images are zero-padded to ensure consistent input sizes.

| Dataset | Description | Download Link |
|--------|-------------|---------------|
| **Original DRIVE** | With FOV masks | https://drive.google.com/file/d/1fNaGhYTQ5dUA1X38U7yhVdYNwiISdWPR/view |
| **Augmented DRIVE** | Pre-processed (592 × 592) | https://drive.google.com/file/d/1t_UxlVWZXBtJQQNxW0vPdwrnqcdYdrRs/view |
| **Augmented STARE** | Pre-processed (704 × 704) | https://drive.google.com/file/d/1-704iIpSQfuVlfKR1p8HUW4cYCDliNLr/view |

**Notes:**

- DRIVE images are zero-padded to **592 × 592**
- STARE images are zero-padded to **704 × 704**
- During testing, predictions are cropped back to original sizes:
  - DRIVE: **584 × 565**
  - STARE: **700 × 605**


## 🚀 Training & Evaluation

### Training Configuration

Following the paper settings:

- **Optimizer:** Adam (learning rate = 1 × 10⁻³)
- **Loss Function:**  
  \[
  \mathcal{L} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{MCC}}
  \]
- **Batch Size:** 8 (DRIVE), 2 (STARE)
- **Epochs:** 150 (with early stopping)
- **Regularization:** DropBlock (rate = 0.15, block size = 7)


## 📊 Results

### Quantitative Results on DRIVE

| Model | F1 Score | Jaccard | Params (M) | GFLOPs |
|------|----------|---------|------------|--------|
| U-Net | 81.30 | 68.52 | 8.64 | 137.10 |
| Attention U-Net | 81.47 | 68.76 | 8.65 | 425.05 |
| SA-UNet | 82.44 | 70.15 | 0.54 | 26.54 |
| **SA-UNetv2 (Ours)** | **82.82** | **70.69** | **0.26** | **21.19** |

### Qualitative Results

Visual comparison on the **DRIVE** and **STARE** datasets demonstrates that **SA-UNetv2** produces more continuous and accurate vessel boundaries, especially for thin vessels and bifurcation regions.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{guo2025saunetv2,
  title   = {SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation},
  author  = {Guo, Changlu and Christensen, Anders Nymark and Dahl, Anders Bjorholm and Yi, Yugen and Hannemose, Morten Rieger},
  journal = {arXiv preprint arXiv:2509.11774},
  year    = {2025}
}



