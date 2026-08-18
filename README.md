# Deep Learning Classification of Tumour–Vessel Contact in Pancreatic CT Images

This repository contains the code developed as part of my MSc dissertation. This project investigates deep learning approaches to classifying tumour-vessel contact in pancreatic CT images.

## Methodology

The project consists of two main parts.

### Part 1: Geometric Label Derivation

Tumour–vessel contact labels were derived from the tumour and vessel segmentation
masks available in the PanTS dataset.

- 3D centreline extraction was performed for the arterial vessel masks.
- A graph-based approach was used to extract the centreline of the continuous
  venous segmentation.
- Tumour–vessel encasement angles were quantified along the vessel centrelines.
- Cases were assigned binary labels (`high_vascular_contact` or `low_vascular_contact`) according to
  whether the measured encasement angle was above or below 180°.

The centreline extraction and encasement-angle methodology for the arterial vessels was adapted from Zhang et al. (2026),
with modifications made for the PanTS dataset and the available vessel segmentations. A separate graph-based approach was
developed for extracting the centreline from the veins segmentation mask.

### Part 2: Deep Learning Classification

The geometrically derived labels were used as targets for deep learning
classification from pancreatic CT images.

- ResNet-50 was used as the CNN architecture.
- Transfer learning from ImageNet-pretrained weights was employed.
- 2D and 2.5D image representations were investigated.
- Classification performance was evaluated using AUROC, AUPRC, precision,
  recall, and F1-score.

## Repository Structure

```
├── notebooks/
│   └── Checks and visualisations performed during the label derivation
│       and classification stages
│
├── src/
│   ├── label_derivation/
│   │   └── Main code for geometric tumour–vessel label derivation
│   │
│   └── classification/ct_only/
│       └── Main code for CT preprocessing, model training, and evaluation
│
├── config.py
│   └── Local dataset paths
│
├── env.sh
│   └── Environment setup for running the code
│
├── requirements.txt
│   └── Python dependencies
│
├── slurm_ct_only_2d
│   └── SLURM script for 2D classification experiments on the Hydra cluster
│
└── slurm_ct_only_2_5d
    └── SLURM script for 2.5D classification experiments on the Hydra cluster
```

## References

### Dataset

W. Li, X. Zhou, Q. Chen, et al., "PanTS: The Pancreatic Tumor Segmentation
Dataset," arXiv:2507.01291 [eess.IV], 2025.
doi: 10.48550/arXiv.2507.01291.

Dataset repository: https://github.com/MrGiovanni/PanTS

### Methodology

Y. Zhang, H. Zhang, Y. Yang, et al., "A clinically validated 3D deep
learning approach for quantifying vascular invasion in pancreatic cancer,"
*npj Digital Medicine*, vol. 9, Art. no. 79, 2026.
doi: 10.1038/s41746-025-02260-3.
