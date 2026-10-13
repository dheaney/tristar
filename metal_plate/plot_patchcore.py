"""Plot the best PatchCore model's confusion matrix on metal_plate.

Run from the repo root:
    python -m metal_plate.plot_patchcore
"""
from patchcore.plot import plot_patchcore

from .datamodule import HERE, make_datamodule

if __name__ == "__main__":
    plot_patchcore(make_datamodule(), out_dir=HERE)
