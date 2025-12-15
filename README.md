<div align="center">

MedBIND3D: 3D Swin Transformer for Brain Tumor Segmentation

Actionable guide to setup, run notebooks, train, infer, and evaluate.

</div>

## Overview

MedBIND3D is a 3D medical image segmentation project focused on multi-class glioma segmentation using hierarchical vision transformers (3D Swin Transformer). The repository includes end-to-end workflows for data preparation, training, inference, evaluation, visualization, and interpretability, primarily driven by Jupyter notebooks with supporting Python stubs under `src/`.

## Repository Structure

- **Main workflows (notebooks):** [notebooks](notebooks)
  - **[ModelSetup.ipynb](notebooks/ModelSetup.ipynb)** — Initialize environment, configure paths, dataset locations, and GPU settings. Establishes global configuration for all downstream notebooks.
  - **[Preprocessing.ipynb](notebooks/Preprocessing.ipynb)** — Load BraTS-2020 NIfTI files, apply skull-stripping validation, perform z-score normalization and intensity clipping, execute center-cropping to 128×128×128, and generate preprocessed volumes ready for training.
  - **[Training.ipynb](notebooks/Training.ipynb)** — Comprehensive training pipeline: data loading, model initialization, loss function setup (Dice + CE), optimizer configuration (AdamW), and epoch-by-epoch training with validation loops. Includes checkpointing and metrics tracking.
  - **[Evaluation.ipynb](notebooks/Evaluation.ipynb)** — Post-training analysis: compute Dice, IoU, Hausdorff-95, sensitivity, specificity, and precision across validation set. Generate statistical summaries and per-class performance breakdowns.
  - **[Week6_InferenceAndReport.ipynb](notebooks/Week6_InferenceAndReport.ipynb)** — Run inference on test/validation sets, post-process predictions (removing disconnected components, morphological operations), generate prediction overlays, and compile final markdown reports with performance tables.
  - **[DataExploration.ipynb](notebooks/DataExploration.ipynb)** — Exploratory data analysis: visualize sample MRI slices across modalities (T1, T1ce, T2, FLAIR), analyze tumor distribution, compute intensity statistics, and document dataset characteristics.
  - **[data_diagnostic.ipynb](notebooks/data_diagnostic.ipynb)** — Diagnostic checks on preprocessed data: validate cropping, verify normalization ranges, check for NaNs/Infs, and confirm batch loading correctness before training.
- **Explainability:** [explainability_maps](explainability_maps)
- **Anatomical factorization:** [Anatomical](Anatomical)
  - [anatomical.ipynb](Anatomical/anatomical.ipynb)
  - [anatomical_factorization.ipynb](Anatomical/anatomical_factorization.ipynb)
- **LOFE experiments:** [LOFE](LOFE)
  - [Week2_LOFE_PK_Encoder_Modality_Factorization.ipynb](LOFE/Week2_LOFE_PK_Encoder_Modality_Factorization.ipynb)
- **MLFM experiments:** [MLFM](MLFM)
  - [MLFM_training.ipynb](MLFM/MLFM_training.ipynb)
  - [figures](MLFM/MLFM/figures), [results](MLFM/MLFM/results)
- **Protocol redundancy:** [Protocol_Redundancy](Protocol_Redundancy)
  - [protocol_redundancy_analysis.ipynb](Protocol_Redundancy/protocol_redundancy_analysis.ipynb)
- **Results and reports:** [results/week5](results/week5)
  - [report_week5.md](results/week5/report_week5.md)
- **Plots & visualizations:** [plots](plots)
  - Samples: [training_curves.png](plots/training_curves.png), [prediction_sample_1.png](plots/prediction_sample_1.png)
  - Week 5: [week5](plots/week5) (Dice plots, heatmaps, overlays)

## Quick Start

### 1) Environment

- Python 3.8+ recommended.
- Create a virtual environment and install dependencies.

```bash
# Create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
# Note: requirements.txt is currently minimal/empty.
# Install per-notebook imports (PyTorch, MONAI, nibabel, SimpleITK, numpy, scipy, scikit-image, pandas, matplotlib, seaborn, plotly).
pip install torch monai nibabel SimpleITK numpy scipy scikit-image pandas matplotlib seaborn plotly
```

### 2) Data Preparation

- Place raw BraTS-2020 MRI volumes in a data directory of your choice.
- Start with [ModelSetup.ipynb](notebooks/ModelSetup.ipynb) to configure paths and environment settings.
- Run [Preprocessing.ipynb](notebooks/Preprocessing.ipynb) to normalize, crop, and prepare all volumes.

### 3) Training

- Execute [Training.ipynb](notebooks/Training.ipynb) to train the 3D Swin Transformer model.
- Training outputs (metrics, curves, checkpoints) are saved to [plots](plots) and [results](results).

### 4) Inference & Reporting

- Use [Week6_InferenceAndReport.ipynb](notebooks/Week6_InferenceAndReport.ipynb) to run inference on validation/test sets and generate summaries.
- Visualizations and overlays are saved under [plots/week5](plots/week5) and HTML interactive views (e.g., [07_interactive_dice_hausdorff.html](plots/week5/07_interactive_dice_hausdorff.html)).

### 5) Evaluation & Explainability

- Evaluate metrics via [Evaluation.ipynb](notebooks/Evaluation.ipynb) and store summaries in [results/week5](results/week5).
- Explore Grad-CAM, attention maps, and factorization analyses via [explainability_maps](explainability_maps), [Anatomical](Anatomical), and [MLFM](MLFM).

## Typical Workflow

1. **Setup:** Configure environment and paths in [ModelSetup.ipynb](notebooks/ModelSetup.ipynb).
2. **Exploration (optional):** Run [DataExploration.ipynb](notebooks/DataExploration.ipynb) and [data_diagnostic.ipynb](notebooks/data_diagnostic.ipynb) to inspect dataset characteristics.
3. **Preprocessing:** Normalize and crop volumes using [Preprocessing.ipynb](notebooks/Preprocessing.ipynb).
4. **Training:** Train model with [Training.ipynb](notebooks/Training.ipynb).
5. **Inference & Reporting:** Generate predictions and reports via [Week6_InferenceAndReport.ipynb](notebooks/Week6_InferenceAndReport.ipynb).
6. **Evaluation:** Compute comprehensive metrics and visualizations with [Evaluation.ipynb](notebooks/Evaluation.ipynb), outputs in [plots](plots) and [results](results).

## Notes on Future `src/` Scripts

When you're ready to move workflows from notebooks into production scripts, you can implement modules for dataset handling, model definition, training loops, inference utilities, and metric computations.

## Results & Reports

- Weekly summary and analyses are documented in [results/week5/report_week5.md](results/week5/report_week5.md).
- Visual performance summaries and overlays are in [plots/week5](plots/week5).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- Built with PyTorch and MONAI for 3D medical imaging.
- Uses common medical imaging libraries (NiBabel, SimpleITK) and scientific Python stack.
