<div align="center">

# 3D Swin Transformer for Brain Tumor Segmentation

### A Deep Learning Framework for Multi-Class Glioma Segmentation Using Hierarchical Vision Transformers

**BraTS-2020 Challenge Implementation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-purple.svg)](https://github.com)

</div>

<div align="justify">

## Project Overview

Brain tumor segmentation remains one of the most challenging problems in medical image analysis due to the high variability in tumor appearance, location, and structure across patients. This project presents a comprehensive implementation of a 3D Swin Transformer architecture for automated brain tumor segmentation on multi-modal MRI scans. The hierarchical architecture progressively reduces spatial resolution while increasing feature dimensionality, enabling the model to learn multi-scale representations essential for accurate tumor boundary delineation. The primary objectives are to develop a robust 3D medical image segmentation pipeline, achieve competitive performance on the BraTS 2020 benchmark across three tumor sub-regions (Enhancing Tumor, Tumor Core, and Whole Tumor), provide comprehensive model interpretability through Grad-CAM and attention visualization, and establish a reproducible framework for transformer-based medical imaging models.

</div>

<div align="center">

## Model Architecture

</div>

<div align="justify">

The 3D Swin Transformer extends the 2D architecture to volumetric medical imaging through three-dimensional shifted windows and hierarchical feature extraction. The input 3D MRI volume with 4 modalities (T1, T1ce, T2, FLAIR) is partitioned into non-overlapping 3D patches of size $4 \times 4 \times 4$, with each patch linearly projected to an embedding dimension of 48. The model employs four hierarchical stages with depths $[2, 2, 2, 2]$ and attention heads $[3, 6, 12, 24]$, progressively downsampling spatial resolution while doubling feature dimensions.

Each Swin Transformer block consists of windowed multi-head self-attention (W-MSA) and shifted window multi-head self-attention (SW-MSA):

$$
\hat{\mathbf{z}}^{l} = \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1}
$$

$$
\mathbf{z}^{l} = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l})) + \hat{\mathbf{z}}^{l}
$$

Attention is computed within local 3D windows of size $4 \times 4 \times 4$ containing 64 patches:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

The decoder progressively upsamples features through transposed 3D convolutions, and the final segmentation head applies a $1 \times 1 \times 1$ convolution to produce 4-class output logits.

</div>

<div align="center">

## Dataset Description

</div>

<div align="justify">

This work utilizes the BraTS 2020 dataset comprising 369 training cases and 125 validation cases. Each patient scan includes four co-registered 3D MRI sequences: T1-weighted (anatomical reference), T1-weighted Contrast-Enhanced (highlighting active tumor), T2-weighted (revealing edema), and FLAIR (edema enhancement with CSF suppression). All sequences are skull-stripped, co-registered, and resampled to $1 \text{mm}^3$ isotropic resolution with dimensions $240 \times 240 \times 155$.

Ground truth annotations delineate three tumor sub-regions: Necrotic Core (Label 1), Peritumoral Edema (Label 2), and Enhancing Tumor (Label 4). For evaluation, these form three hierarchical regions:

$$
\begin{align}
\text{Enhancing Tumor (ET)} &= \{\text{Label } 4\} \\
\text{Tumor Core (TC)} &= \{\text{Label } 1, \text{Label } 4\} \\
\text{Whole Tumor (WT)} &= \{\text{Label } 1, \text{Label } 2, \text{Label } 4\}
\end{align}
$$

All volumes undergo z-score normalization, intensity clipping to $[-5, 5]$, and center-cropping to $128 \times 128 \times 128$ for efficient batch processing.

</div>

<div align="center">

## Implementation Pipeline

</div>

<div align="justify">

The model is optimized using a composite loss combining Dice loss and cross-entropy:

$$
\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{CE}}
$$

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{K} \sum_{k=1}^{K} \frac{2 \sum_{i} p_{i,k} g_{i,k}}{\sum_{i} p_{i,k} + \sum_{i} g_{i,k} + \epsilon}
$$

Training employs AdamW optimizer with initial learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-5}$, batch size 2, and 100 epochs with cosine annealing. Mixed precision training (FP16) reduces memory consumption and accelerates computation.

</div>

<div align="center">

## Mathematical Formulations

</div>

<div align="justify">

### Key Segmentation Metrics

**Dice Similarity Coefficient:**

$$
\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \text{TP}}{2 \text{TP} + \text{FP} + \text{FN}}
$$

**Intersection over Union:**

$$
\text{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

**Hausdorff Distance (95th percentile):**

$$
\text{HD}_{95}(P, G) = \max\left(\text{percentile}_{95}(d(P, G)), \text{percentile}_{95}(d(G, P))\right)
$$

**Sensitivity and Specificity:**

$$
\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}
$$

</div>

<div align="center">

## Evaluation Results

</div>

<div align="justify">

The 3D Swin Transformer was evaluated on the BraTS 2020 validation set (125 cases). Performance metrics are reported as mean ± standard deviation:

| **Metric** | **Enhancing Tumor (ET)** | **Tumor Core (TC)** | **Whole Tumor (WT)** |
|:-----------|:------------------------:|:-------------------:|:--------------------:|
| **Dice Coefficient** | 0.7823 ± 0.1456 | 0.8512 ± 0.1203 | 0.9045 ± 0.0867 |
| **IoU (Jaccard)** | 0.6734 ± 0.1589 | 0.7623 ± 0.1398 | 0.8345 ± 0.1045 |
| **Hausdorff Distance (95%)** | 4.32 ± 6.78 mm | 6.89 ± 8.45 mm | 5.67 ± 7.23 mm |
| **Sensitivity** | 0.8156 ± 0.1523 | 0.8734 ± 0.1089 | 0.9234 ± 0.0756 |
| **Specificity** | 0.9987 ± 0.0012 | 0.9978 ± 0.0015 | 0.9956 ± 0.0023 |
| **Precision** | 0.7689 ± 0.1678 | 0.8423 ± 0.1267 | 0.8923 ± 0.0934 |

The model achieves strongest performance on Whole Tumor segmentation (Dice: 0.9045), reflecting high contrast between tumor and healthy tissue. Tumor Core segmentation achieves competitive performance (Dice: 0.8512), while Enhancing Tumor segmentation is most challenging (Dice: 0.7823) due to smaller spatial extent and irregular boundaries. The exceptionally high specificity (>0.995) demonstrates strong ability to avoid false positives in healthy tissue.

</div>

<div align="center">

## Explainability and Visualization

</div>

<div align="justify">

### Gradient-weighted Class Activation Mapping (Grad-CAM)

Grad-CAM generates visual explanations by computing gradients of class scores with respect to feature maps:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i,j,k} \frac{\partial y^c}{\partial A_{ijk}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$

Visualizations reveal that the model correctly focuses on hyperintense regions in T1ce for Enhancing Tumor, attends to both necrotic centers and enhancement for Tumor Core, and captures the entire extent of signal abnormality across T2/FLAIR for Whole Tumor.

### Attention Map Analysis

Attention patterns demonstrate hierarchical processing: early layers focus on local textures and edges within $4 \times 4 \times 4$ windows, middle layers integrate information across broader regions through shifted windowing, and late layers exhibit global semantic patterns connecting spatially distant locations with similar tissue characteristics. Feature activation visualization across four stages shows progressive abstraction from low-level intensity patterns to high-level semantic tumor representations.

### 3D Heatmap Animations

Animated GIF visualizations across all slices reveal spatial consistency in attention patterns, with smooth transitions between adjacent slices confirming robust 3D feature learning. Heatmap intensity is consistently highest within tumor boundaries and decreases gradually in surrounding tissue.

</div>

<div align="center">

## Weekly Progress Summaries

</div>

<div align="justify">

**Week 1: Dataset Preparation** — Downloaded and organized BraTS 2020 dataset (369 training, 125 validation cases), implemented data loading pipelines, conducted exploratory analysis of tumor distributions and intensity statistics, developed preprocessing scripts for normalization and cropping, and established train/validation splits.

**Week 2: Architecture Implementation** — Implemented complete 3D Swin Transformer including patch embedding, windowed attention, shifted windows, and hierarchical stages; integrated mixed precision training; configured AdamW optimizer with cosine annealing; implemented Dice-CE composite loss; achieved initial validation Dice of 0.75 for WT after 30 epochs.

**Week 3: Hyperparameter Optimization** — Conducted grid search over learning rates (optimal: $10^{-4}$), ablation studies on Dice-CE weighting (optimal: $\alpha = 0.5$), experimented with window sizes (optimal: $4^3$), added elastic deformations and intensity perturbations for augmentation, implemented early stopping; achieved validation Dice of 0.78 (ET), 0.85 (TC), 0.90 (WT).

**Week 4: Advanced Training** — Implemented deep supervision with auxiliary heads, integrated test-time augmentation with 8 variants, adopted exponential moving average of weights, extended training to 100 epochs with warm restarts, implemented gradient clipping; achieved peak validation Dice of 0.782 (ET), 0.851 (TC), 0.905 (WT).

**Week 5: Evaluation & Explainability** — Computed comprehensive metrics (Dice, IoU, Hausdorff, Sensitivity, Specificity, Precision) for 125 validation cases; generated 10+ visualization types (boxplots, violin plots, radar charts, correlation heatmaps, 3D overlays); implemented Grad-CAM for 3D, extracted attention maps from Swin blocks, visualized feature activations across layers, created 3D heatmap GIF animations; saved all metrics to CSV and generated detailed markdown report.

</div>

<div align="center">

## Outputs Generated

</div>

<div align="justify">

**Metrics:** `metrics_week5.csv` containing comprehensive evaluation metrics for all 125 validation samples (Dice per class, Hausdorff distances, sensitivity, specificity, precision, recall, IoU, volumetric overlap error, tumor volumes).

**Visualizations (15+ plots):** Dice boxplots per class, Dice violin plots with smooth distributions, radar charts comparing ET/TC/WT performance, correlation heatmap between metrics, bar charts with mean Dice and standard deviation, volume distribution histograms (GT vs Pred), performance matrix (6 metrics × 3 regions), best/worst case tables, 3D tumor overlay visualizations (axial/coronal/sagittal views), interactive Plotly scatter plots with tooltips.

**Explainability Maps:** Grad-CAM heatmaps for all tumor classes (Necrotic, Edema, Enhancing), attention map visualizations from early and late Swin Transformer layers, feature activation plots across 4 hierarchical stages, 3D heatmap GIF animations showing slice-by-slice activation patterns.

**Reports:** `report_week5.md` containing performance summary tables, best/worst case analysis, explainability findings, conclusions, and complete file inventory.

</div>

<div align="center">

## Conclusions

</div>

<div align="justify">

This work demonstrates that 3D Swin Transformers achieve competitive performance on brain tumor segmentation through hierarchical feature learning and shifted window attention mechanisms. The model achieves an overall mean tumor Dice of 0.846 across all three regions, with Whole Tumor (WT) as the best performing region (Dice: 0.9045). Explainability analysis confirms the model learns meaningful tumor features, with attention patterns aligning with radiological characteristics of gliomas. The hierarchical architecture successfully captures multi-scale representations, with early layers extracting local textures and late layers encoding global semantic patterns.

Future research directions include: (1) exploring larger model capacities with more Swin Transformer blocks, (2) incorporating uncertainty quantification through Bayesian deep learning or ensemble methods, (3) extending to multi-task learning with survival prediction, (4) investigating self-supervised pre-training on large unlabeled MRI datasets, (5) developing efficient inference strategies for real-time clinical deployment, and (6) validating model generalization across multi-institutional datasets beyond BraTS.

</div>

<div align="center">

## Tech Stack

</div>

<div align="justify">

**Deep Learning Framework:** PyTorch 2.0+, CUDA 11.8, cuDNN 8.6

**Medical Imaging:** MONAI, NiBabel, SimpleITK, MedPy

**Scientific Computing:** NumPy, SciPy, scikit-image, pandas

**Visualization:** Matplotlib, Seaborn, Plotly, imageio

**Development Tools:** Jupyter Notebook, Python 3.8+, Git

**Hardware:** NVIDIA RTX 3090 (24GB VRAM), 64GB RAM, Intel i9 CPU

</div>


<div align="center">

## License

This project is licensed under the MIT License.

</div>