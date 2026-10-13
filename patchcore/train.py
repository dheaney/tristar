"""Grid search over PatchCore hyperparameters for a given datamodule.

Writes <out_dir>/patchcore_grid.csv and saves the best model to
<out_dir>/best_model/{best_model.ckpt,best_config.json}.
"""
import os
import gc, time, json
import pandas as pd
import torch

from anomalib.models import Patchcore
from anomalib.engine import Engine

layer_options = [
    ("layer2",),
    ("layer3",),
    ("layer2", "layer3"),  # default from PatchCore
]
# coreset ratio affects fit (memory bank size), so each value needs its own fit.
# num_neighbors only matters at inference, so it is swept without refitting.
coreset_options = [0.01, 0.1, 0.25, 0.5]  # 0.01 = PatchCore paper default
neighbor_options = [1, 3, 5, 9, 15]

# metrics used to pick the best model, in priority order (later keys break ties)
SCORE_KEYS = ("image_AUROC", "image_F1Score")
WORST = tuple(-float("inf") for _ in SCORE_KEYS)


def _score(res):
    return tuple(res.get(k, -float("inf")) for k in SCORE_KEYS)


def train_patchcore(datamodule, out_dir="."):
    best_dir = os.path.join(out_dir, "best_model")
    grid_path = os.path.join(out_dir, "patchcore_grid.csv")
    os.makedirs(best_dir, exist_ok=True)

    best = {"score": WORST}

    rows = []
    for layers in layer_options:
        name = "+".join(layers)
        for ratio in coreset_options:
            model = engine = None
            try:
                model = Patchcore(
                    backbone="wide_resnet50_2",
                    layers=layers,
                    coreset_sampling_ratio=ratio,
                    num_neighbors=neighbor_options[0],
                )
                engine = Engine(enable_progress_bar=False)

                t0 = time.time()
                engine.fit(model=model, datamodule=datamodule)
                fit_s = time.time() - t0

                cfg_best = {"score": WORST}
                for k in neighbor_options:
                    model.model.num_neighbors = k
                    t0 = time.time()
                    res = engine.test(model=model, datamodule=datamodule)[0]
                    row = {
                        "layers": name,
                        "coreset_sampling_ratio": ratio,
                        "num_neighbors": k,
                        "fit_s": round(fit_s, 1),
                        "test_s": round(time.time() - t0, 1),
                        **res,
                    }
                    rows.append(row)
                    print(row)

                    score = _score(res)
                    if score > cfg_best["score"]:
                        cfg_best = {"score": score, "k": k, "row": row}

                # Save the model once per config, only if it beats the global best,
                # using that config's best num_neighbors.
                if cfg_best["score"] > best["score"]:
                    k = cfg_best["k"]
                    model.model.num_neighbors = k
                    tag = f"{name}_cs{ratio}_k{k}"
                    engine.trainer.save_checkpoint(os.path.join(best_dir, "best_model.ckpt"))
                    with open(os.path.join(best_dir, "best_config.json"), "w") as f:
                        json.dump(
                            {
                                "layers": list(layers),
                                "coreset_sampling_ratio": ratio,
                                "num_neighbors": k,
                                "backbone": "wide_resnet50_2",
                                "metrics": {kk: float(v) for kk, v in cfg_best["row"].items()
                                            if isinstance(v, (int, float)) and kk.startswith(("image_", "pixel_"))},
                            },
                            f, indent=2,
                        )
                    best = cfg_best
                    scores = ", ".join(f"{k}={v:.4f}" for k, v in zip(SCORE_KEYS, best["score"]))
                    print(f"New best: {tag} ({scores}) -> saved to {best_dir}/")

            except torch.cuda.OutOfMemoryError:
                print(f"OOM on {name}, ratio={ratio}, skipping")
                rows.append({"layers": name, "coreset_sampling_ratio": ratio,
                             "num_neighbors": None, "error": "OOM"})

            finally:
                pd.DataFrame(rows).to_csv(grid_path, index=False)  # save progress
                del model, engine
                gc.collect()
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    sort_keys = [k for k in SCORE_KEYS if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, ascending=False, na_position="last")
    print(df.to_string(index=False))
    return df
