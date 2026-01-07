# CSIRO Biomass Prediction using DINOv3

This repository contains a metric-optimized machine learning pipeline
for the CSIRO Biomass Estimation Kaggle Competition.

## Highlights
- DINOv3 ConvNeXt-L feature extraction
- 448x448 high-resolution images
- Test-Time Augmentation (Horizontal Flip)
- Metric-aware target transformations
- Ridge + SVR + LightGBM ensemble

## How to Run
Upload this repo to Kaggle and run:
```bash
python dinov3_biomass.py
```

Generates `submission.csv`.
