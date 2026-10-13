import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import pandas as pd
from pathlib import Path

plots = Path("plots")
plots.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('metal_plate.csv')

X = df[['r_q05', 'b_q05', 'r_std', 'b_std']]
y = df["defect"]


# Stratified split keeps class proportions the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap="Blues",
    ax=ax,
)
ax.set_title("SVM (RBF) Confusion Matrix")
fig.savefig(plots / "svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
