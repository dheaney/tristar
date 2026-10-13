import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    root = Path(yaml.safe_load(f)["metal_plate"]["root"])

def vectorize_image(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    px = img.reshape(-1, 3)
    qs = np.percentile(px, [5, 10, 25, 50, 75, 90, 95], axis=0)
    return np.vstack([
        qs,
        px.mean(axis=0),
        px.std(axis=0)
    ])

def generate_df():
    rows = []
    for split in ("test", "train"):
        for path in (root / split).rglob("*.png"):
            _, defect, image = path.parts[-3:]
            row = {
                "split": split,
                "defect": defect,
                "image": image,
            }
            vector = vectorize_image(path)
            for channel, i in zip("rgb", range(3)):
                row[channel + "_q05"] = vector[0, i]
                row[channel + "_q10"] = vector[1, i]
                row[channel + "_q25"] = vector[2, i]
                row[channel + "_q50"] = vector[3, i]
                row[channel + "_q75"] = vector[4, i]
                row[channel + "_q90"] = vector[5, i]
                row[channel + "_q95"] = vector[6, i]
                row[channel + "_mean"] = vector[7, i]
                row[channel + "_std"] = vector[8, i]
            
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    new_df = generate_df()
    new_df.to_csv("metal_plate.csv")
