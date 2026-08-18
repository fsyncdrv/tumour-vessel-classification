# Deep Learning Classification of Tumour–Vessel Contact in Pancreatic CT Images

## Methodology

The project consists of two main parts.

### Part 1: Geometric Label Derivation

Tumour–vessel contact labels were derived from the tumour and vessel segmentation
masks available in the PanTS dataset.

- 3D centreline extraction was performed for the arterial vessel masks (SMA, CA,
  postcava, and aorta).
- A graph-based approach was used to extract the centreline of the continuous
  venous segmentation.
- Tumour–vessel encasement angles were quantified along the vessel centrelines.
- Cases were assigned binary labels according to a 180° encasement threshold.

The centreline extraction and encasement-angle methodology was adapted from
Zhang et al. (2026), with modifications made for the PanTS dataset and the
available vessel segmentations.

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
│   └── Dataset paths, derived-data paths, and output directory configuration
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


