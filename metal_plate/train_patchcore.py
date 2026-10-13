"""Grid-search PatchCore on metal_plate.

Run from the repo root:
    python -m metal_plate.train_patchcore
"""
from patchcore.train import train_patchcore

from .datamodule import HERE, make_datamodule

if __name__ == "__main__":
    train_patchcore(make_datamodule(), out_dir=HERE)
