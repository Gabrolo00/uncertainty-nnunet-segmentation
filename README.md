# Uncertainty Estimation and Calibration for nnU-Net Ensembles

Voxel-wise uncertainty maps for 3D CT segmentation, used as tools to locate segmentation errors. The project builds ensembles of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) models on public [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) data, calibrates them, computes and evaluates 14+ uncertainty measures, tests everything on an external clinical dataset, and explores new ways to separate the *expected* uncertainty along the organ boundary from the *unexpected* uncertainty that signals real errors.

Master's thesis *Evaluating Uncertainty in Medical Image Segmentation Models Trained on a Public Dataset* by **Gabriele Rolo**, MSc in Computer Science and Engineering, Politecnico di Milano, A.Y. 2025-26. Supervisor: Prof. Daniele Loiacono. The project continues previous work on uncertainty for lymph node segmentation (see [Previous work](#previous-work)).

## What is in this repository

- **End-to-end pipeline**: dataset construction, nnU-Net training (deep ensemble, checkpoint ensemble with cyclical learning rate, fine-tuning from pretrained weights), inference with saved logits, temperature scaling, uncertainty maps, distance-aware measures (DEU, EBCU), leave-one-out threshold calibration, evaluation and statistics.
- **Two target structures**: kidney and trachea from TotalSegmentator, plus an **external test set** of radiotherapy planning CTs from a different hospital (kidney).
- **Benchmark** of uncertainty measures near and away from the boundary, across structures, ensemble strategies and domains.
- **New methods**: combined uncertainty maps and an **autoencoder** that learns the expected uncertainty map.
- **Two interactive viewers** (Streamlit) to inspect segmentations, errors and uncertainty maps.

## Main results

| Question | Finding |
|---|---|
| Deep vs checkpoint ensemble | The deep ensemble is more accurate (kidney Dice 0.937 vs 0.924, trachea 0.969 vs 0.939) and better calibrated; checkpoint members are more diverse but individually less accurate. |
| Calibration | All models are overconfident (T\* = 2.1-3.0). Temperature scaling reduces the NLL by about 46-49% without changing the segmentation, and still works on the external data (-49%). |
| Fine-tuning from TotalSegmentator weights | No gain on external data (Dice 0.851 vs 0.850); fewer diverse members, but their disagreement is better focused on errors. |
| Domain shift | Kidney Dice 0.937 → 0.851 on the external set, largely due to a different annotation convention (renal sinus). The temperature transfers, the error-detection thresholds do not. |
| Best measures | Near the boundary: entropy-based measures and DEU. Away from it: disagreement among members (variance, mutual information). EBCU moves alarms away from the boundary but only pays off with a large review budget. |
| Autoencoder | Residual maps do not beat existing measures: the autoencoder also reconstructs unusual patterns, so the residual is small exactly where uncertainty is unexpected. |

Details and numbers: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Repository structure

```
1_data/              dataset construction (TotalSegmentator) and external test set preparation
2_models/training/   nnU-Net training: deep ensemble, checkpoint ensemble, fine-tuning
2_models/inference/  ensemble inference with saved logits (modified nnU-Net predictor)
3_uncertainty/       temperature/  maps/  distance_aware/  combined/  thresholds/  evaluation/
                     + pipeline_*.sh launchers that run the whole chain for one model
4_analysis/          statistics, model/structure/domain comparisons, external test evaluation, figures/
5_autoencoder/       autoencoder on uncertainty maps (launchers numbered in execution order)
6_viewer/            Streamlit viewers
docs/                PIPELINE.md (full workflow) and EXPERIMENTS.md (experiments and results)
```

Every script starts with a short description and a usage line. The documents in `docs/` and the comments in the code are in Italian.

## The pipeline in short

```
TotalSegmentator ─► binary labels + split ─► nnU-Net training ─► inference (logits of ensemble and members)
                                                                     │
      ┌──────────────────────────────────────────────────────────────┘
      ▼
temperature scaling (LOO on out-of-fold cases) ─► uncertainty maps ─► DEU / EBCU ─► LOO thresholds
      ─► evaluation (15 mm ROI and no-border region) ─► statistics and comparisons ─► viewer
```

Inference is the only expensive step: it is run once and saves the logits; all the analysis works on the saved files.

## Getting started

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
nnU-Net (`nnunetv2==2.6.2`) runs in a separate environment for preprocessing, training and inference. The full workflow, with every script, launcher, environment and convention, is in [`docs/PIPELINE.md`](docs/PIPELINE.md).

Examples:
```bash
# uncertainty maps from saved logits, with the calibrated temperature
python 3_uncertainty/maps/total_uncertainty_measures.py --input_dir <logits_dir> --output_dir <maps_dir> --T <T*>

# viewer
streamlit run 6_viewer/uncertainty_viewer.py
```

## Data and models

No data, model weights or results are included. TotalSegmentator is publicly available; the external clinical dataset is private and cannot be shared. Scripts and launchers use the absolute paths of the server where the project was developed: to run them elsewhere, change the path constants at the top of each script or launcher.

## For students continuing this work

1. Read [`docs/PIPELINE.md`](docs/PIPELINE.md) for the workflow and the conventions (region of interest, no-border region, leave-one-out thresholds, external test without recalibration).
2. Read [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for what has already been tested and what did not work.
3. On the original server, weights, logits, maps and results are already computed under `/mnt/sda2/grolo/`: start from the viewers and the saved results before re-running anything.
4. Open directions: measures that use information beyond the ensemble output (e.g. network features), recalibrating thresholds on a few local cases for a new hospital, more external test sets and structures.

## Previous work

This project extends earlier theses on uncertainty estimation for lymph node segmentation with nnU-Net:
- Lorenzo Mondo: first version of the pipeline and viewer;
- Lorenzo Reitani, *Calibration and Uncertainty Visualisation in Medical Image Segmentation* (2026): ROI-aware temperature scaling, distance-aware measures DEU and EBCU, threshold calibration.

## References

- F. Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods, 2021.
- J. Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*, Radiology: Artificial Intelligence, 2023.
- Y. Zhao et al., *Efficient Bayesian Uncertainty Estimation for nnU-Net*, MICCAI 2022 (checkpoint ensembles with cyclical learning rate).
- B. Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017.

## License

[Apache License 2.0](LICENSE). `2_models/inference/predict_from_raw_data.py` is a modified file from nnU-Net v2 (Apache 2.0); see [`NOTICE`](NOTICE).
