<div align="center">

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

## License

This project is licensed under the MIT License.

</div>