"""
===========================================================
CLEAN HORMONE → ARCHETYPE PREDICTION
===========================================================

GOAL
-----
Predict BEHAVIOR-DERIVED archetypes using ONLY:
- hormones
- metabolites
- neurochemical variables

NO behavioral leakage allowed.

This script:
1. Loads merged data
2. Removes leakage variables
3. Creates hard archetype labels
4. Preprocesses hormone data
5. PCA on hormones
6. Logistic regression classifier
7. Leave-one-out cross-validation
8. Balanced accuracy
9. Confusion matrix
10. Saves results

===========================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

FILE = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Merged_Hormone_Behavior_Data.xlsx"

df = pd.read_excel(FILE)

print(df.shape)

# ============================================================
# KEEP ONLY NUMERIC
# ============================================================

num = df.select_dtypes(include=np.number)

# ============================================================
# TARGETS
# ============================================================

# archetype probabilities

target_cols = [
    c for c in num.columns
    if "Probability_A" in c
]

print("\nTarget columns:")
print(target_cols)

# probability matrix
Y_prob = num[target_cols]

# HARD LABELS
# dominant archetype

y = np.argmax(
    Y_prob.values,
    axis=1
)

# ============================================================
# PREDICTORS
# ============================================================

X_df = num.drop(
    columns=target_cols
)

# ============================================================
# REMOVE LEAKAGE VARIABLES
# ============================================================

remove_keywords = [

    "glicko",

    "behavior",

    "archetype",

    "probability",

    "pc",

    "rank",

    "assigned"
]

drop_cols = []

for c in X_df.columns:

    cl = c.lower()

    # remove behavior-derived variables
    if any(
        k in cl
        for k in remove_keywords
    ):

        drop_cols.append(c)

    # remove IDs
    if "animal" in cl:

        drop_cols.append(c)

    if "mouse" in cl:

        drop_cols.append(c)

# drop leakage
X_df = X_df.drop(
    columns=list(set(drop_cols)),
    errors="ignore"
)

print("\nRemoved columns:")
print(drop_cols)

print("\nRemaining predictors:")
print(X_df.columns.tolist())

# ============================================================
# PREPROCESS
# ============================================================

# impute
imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X_df)

# scale
scaler = StandardScaler()

X = scaler.fit_transform(X)

# ============================================================
# PCA ON HORMONES
# ============================================================

# reduce dimensionality

n_components = min(
    5,
    X.shape[1],
    X.shape[0] - 1
)

pca = PCA(
    n_components=n_components
)

X_pca = pca.fit_transform(X)

print("\nExplained variance:")
print(
    pca.explained_variance_ratio_
)

# ============================================================
# CLASSIFIER
# ============================================================

loo = LeaveOneOut()

pred = np.zeros_like(y)

# ============================================================
# LEAVE-ONE-OUT CROSS VALIDATION
# ============================================================

for train_idx, test_idx in loo.split(X_pca):

    X_train = X_pca[train_idx]
    X_test = X_pca[test_idx]

    y_train = y[train_idx]

    # classifier
    model = LogisticRegression(

        max_iter=5000,

        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    pred[test_idx] = model.predict(
        X_test
    )

# ============================================================
# METRICS
# ============================================================

bal_acc = balanced_accuracy_score(
    y,
    pred
)

print("\n================================================")
print("BALANCED ACCURACY")
print("================================================")

print(
    round(bal_acc, 3)
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    pred
)

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

print(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        y,
        pred
    )
)

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(5,4))

plt.imshow(cm)

plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title(
    f"Confusion Matrix\nBalanced accuracy={bal_acc:.3f}"
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()

# ============================================================
# TOP HORMONE LOADINGS
# ============================================================

loadings = pd.DataFrame(

    pca.components_.T,

    index=X_df.columns,

    columns=[
        f"PC{i+1}"
        for i in range(
            pca.components_.shape[0]
        )
    ]
)

print("\n================================================")
print("TOP HORMONE LOADINGS")
print("================================================")

for pc in loadings.columns[:3]:

    print(f"\n{pc}")

    vals = (
        loadings[pc]
        .abs()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print(vals)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "True_Label": y,

    "Predicted_Label": pred
})

results.to_excel(
    "Hormone_Archetype_Prediction.xlsx",
    index=False
)

print(
    "\nSaved: Hormone_Archetype_Prediction.xlsx"
)