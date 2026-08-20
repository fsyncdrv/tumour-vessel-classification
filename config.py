## I have used an external hard drive to store the PanTS dataset
## and mapped the folders to the original dataset structure
## Full dataset can be accessed on: https://github.com/MrGiovanni/PanTS

from pathlib import Path

PANTS_ROOT = Path("/Volumes/BackupDrive/pants/data")

IMAGE_DIR = PANTS_ROOT / "ImageTr"
IMAGE_TE_DIR = PANTS_ROOT / "ImageTe"
LABEL_DIR = PANTS_ROOT / "LabelTr"
ISO_RESAMPLING_DIR = PANTS_ROOT / "Derived" / "IsoResLabelTr"
DERIVED_ANGLES_DIR = PANTS_ROOT / "Derived" / "DerivedAngles"
OUTPUT_DIR = Path("outputs")
