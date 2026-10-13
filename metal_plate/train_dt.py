import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, cross_val_predict, GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import pandas as pd
import seaborn as sns

plots = Path("plots")
plots.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('metal_plate.csv')

X = df[['r_q05', 'b_q05', 'r_std', 'b_std']]
y = df["defect"]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

scores = cross_validate(
    clf, X, y, cv=cv,
    scoring=["accuracy", "balanced_accuracy", "f1_macro"],
)

for name in ["accuracy", "balanced_accuracy", "f1_macro"]:
    s = scores[f"test_{name}"]
    print(f"{name:18s} {s.mean():.3f} ± {s.std():.3f}  (min {s.min():.3f}, max {s.max():.3f})")

oof_pred = cross_val_predict(
    clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, oof_pred, cmap="Blues", ax=ax)
ax.set_title("Decision Tree: Out-of-Fold Confusion Matrix")
fig.savefig(plots / "dt_confusion_matrix_oof.png", dpi=300, bbox_inches="tight")
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [2, 3, 4, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="balanced_accuracy",
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_, "| CV score:", round(grid.best_score_, 3))

# Plot grid search results
res = pd.DataFrame(grid.cv_results_)
res["max_depth"] = res["param_max_depth"].astype(object).fillna("None").astype(str)
res["min_samples_leaf"] = res["param_min_samples_leaf"].astype(int)

depth_order = ["2", "3", "4", "5", "7", "10", "None"]

mean = res.pivot(index="min_samples_leaf", columns="max_depth", values="mean_test_score")[depth_order]
std = res.pivot(index="min_samples_leaf", columns="max_depth", values="std_test_score")[depth_order]

# Annotate each cell with mean +/- std
annot = mean.round(3).astype(str) + "\n±" + std.round(3).astype(str)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(mean, annot=annot, fmt="", cmap="viridis", cbar_kws={"label": "CV balanced accuracy"}, ax=ax)

# Outline the best cell
bi = list(mean.index).index(grid.best_params_["min_samples_leaf"])
bj = depth_order.index(str(grid.best_params_["max_depth"]))
ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False, edgecolor="red", lw=3))

ax.set_title("Decision tree grid search (5-fold stratified CV)")
fig.savefig(plots / "dt_grid_search.png", dpi=300, bbox_inches="tight")
plt.close(fig)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
ax.set_title("Tuned Tree: Held-Out Test Set")
fig.savefig(plots / "dt_confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.close(fig)
