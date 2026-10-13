from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('metal_plate.csv')

out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)

stats = ["q05", "q10", "q25", "q50", "q75", "q90", "q95", "mean", "std"]
labels = {
    "q05": "5th percentile", "q10": "10th percentile", "q25": "25th percentile",
    "q50": "Median", "q75": "75th percentile", "q90": "90th percentile",
    "q95": "95th percentile", "mean": "Mean", "std": "Std dev",
}
order = ["R", "G", "B"]
palette = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}

for stat in stats:
    long = (
        df.melt(
            id_vars="defect",
            value_vars=[f"{c}_{stat}" for c in "rgb"],
            var_name="channel",
            value_name=stat,
        )
        .assign(channel=lambda d: d["channel"].str[0].str.upper())
    )

    g = sns.catplot(
        data=long,
        kind="violin",
        x="channel",
        y=stat,
        hue="channel",
        order=order,
        hue_order=order,
        palette=palette,
        col="defect",
        col_wrap=4,
        inner="quartile",
        cut=0,
        legend=False,
        height=3,
        aspect=0.8,
    )
    g.set_axis_labels("Channel", labels[stat])
    g.set_titles("{col_name}")
    g.figure.suptitle(f"{labels[stat]} by channel and defect")

    g.savefig(out_dir / f"violin_{stat}_by_defect.png", dpi=200, bbox_inches="tight")
    plt.close(g.figure)
