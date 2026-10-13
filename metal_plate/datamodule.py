"""Dataset config for metal_plate, shared by its train/plot scripts.

Paths are read from config.yaml at the repo root.
"""
import os

import yaml
from anomalib.data import Folder

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "..", "config.yaml")


def make_datamodule():
    with open(CONFIG_PATH) as f:
        paths = yaml.safe_load(f)["metal_plate"]
    return Folder(
        name="metal_plate",
        root=paths["root"],
        normal_dir=paths["normal_dir"],
        abnormal_dir=paths["abnormal_dir"],
        normal_test_dir=paths["normal_test_dir"],
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        seed=42
    )
