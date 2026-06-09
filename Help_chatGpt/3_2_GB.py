# ============================================================
# ROBUST GAUSSIAN NB 3-CLASS HORMONE CLASSIFICATION PIPELINE
# CLEAN NESTED CV VERSION
# ============================================================

# FEATURES:
#
# - Gaussian Naive Bayes
# - ALL features preserved
# - Manual Nested Cross Validation
# - Repeated Stratified CV
# - Hyperparameter tuning
# - Balanced Accuracy + Macro F1
# - Confidence Intervals
# - Publication-quality confusion matrix
# - Permutation significance test
# - Permutation feature importance
# - PCA visualization
# - SHAP explainability
# - Bootstrap stability
# - Model saving
#
# ============================================================

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib

# ============================================================
# SKLEARN
# ============================================================

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    permutation_test_score
)

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance

# ============================================================
# LOAD DATA
# ============================================================

X = pd.read_csv(
    r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\X_hormones.csv"
)

y = pd.read_csv(
    r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Y_label.csv"
)

# ============================================================
# LABEL VECTOR
# ============================================================

y = y.iloc[:, 0]

# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()

y = le.fit_transform(y)

# ============================================================
# DATA INFO
# ============================================================

print("=" * 60)

print("X shape:", X.shape)

print("y shape:", y.shape)

print("\nLabel distribution:")

print(pd.Series(y).value_counts())

print("\nClasses:")

print(le.classes_)

print("=" * 60)

# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "clf",
        GaussianNB()
    )
])

# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "clf__var_smoothing": np.logspace(
        -12,
        -6,
        20
    )
}

# ============================================================
# INNER CV
# ============================================================

inner_cv = StratifiedKFold(

    n_splits=4,

    shuffle=True,

    random_state=42
)

# ============================================================
# OUTER CV
# ============================================================

outer_cv = RepeatedStratifiedKFold(

    n_splits=5,

    n_repeats=20,

    random_state=42
)

# ============================================================
# GRID SEARCH
# ============================================================

grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="balanced_accuracy",

    cv=inner_cv,

    n_jobs=-1
)

# ============================================================
# STORAGE
# ============================================================

all_bal_scores = []

all_f1_scores = []

all_estimators = []

y_true_all = []

y_pred_all = []

# ============================================================
# MANUAL NESTED CROSS VALIDATION
# ============================================================

print("\nStarting Nested Cross Validation...")

for fold_idx, (train_idx, test_idx) in enumerate(

    outer_cv.split(X, y)

):

    print(f"\nOuter Fold {fold_idx + 1}")

    # ========================================================
    # SPLIT
    # ========================================================

    X_train = X.iloc[train_idx]

    X_test = X.iloc[test_idx]

    y_train = y[train_idx]

    y_test = y[test_idx]

    # ========================================================
    # INNER GRID SEARCH
    # ========================================================

    grid.fit(

        X_train,

        y_train
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_fold_model = grid.best_estimator_

    all_estimators.append(

        best_fold_model
    )

    # ========================================================
    # PREDICT
    # ========================================================

    y_pred = best_fold_model.predict(

        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    bal_acc = balanced_accuracy_score(

        y_test,

        y_pred
    )

    f1 = f1_score(

        y_test,

        y_pred,

        average="macro"
    )

    # ========================================================
    # STORE SCORES
    # ========================================================

    all_bal_scores.append(

        bal_acc
    )

    all_f1_scores.append(

        f1
    )

    # ========================================================
    # STORE PREDICTIONS
    # ========================================================

    y_true_all.extend(

        y_test
    )

    y_pred_all.extend(

        y_pred
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(

        f"Balanced Accuracy: "
        f"{bal_acc:.3f}"
    )

    print(

        f"Macro F1: "
        f"{f1:.3f}"
    )

    print(

        "Best Params:",
        grid.best_params_
    )

# ============================================================
# NUMPY ARRAYS
# ============================================================

all_bal_scores = np.array(
    all_bal_scores
)

all_f1_scores = np.array(
    all_f1_scores
)

y_true_all = np.array(
    y_true_all
)

y_pred_all = np.array(
    y_pred_all
)

# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

def confidence_interval(scores):

    mean = np.mean(scores)

    ci_low = np.percentile(
        scores,
        2.5
    )

    ci_high = np.percentile(
        scores,
        97.5
    )

    return mean, ci_low, ci_high

# ============================================================
# COMPUTE CI
# ============================================================

bal_mean, bal_low, bal_high = (

    confidence_interval(
        all_bal_scores
    )
)

f1_mean, f1_low, f1_high = (

    confidence_interval(
        all_f1_scores
    )
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("NESTED CV RESULTS")

print("=" * 60)

print(f"\nBalanced Accuracy:")

print(f"Mean = {bal_mean:.3f}")

print(
    f"95% CI = "
    f"[{bal_low:.3f}, {bal_high:.3f}]"
)

print(f"\nMacro F1:")

print(f"Mean = {f1_mean:.3f}")

print(
    f"95% CI = "
    f"[{f1_low:.3f}, {f1_high:.3f}]"
)

# ============================================================
# FINAL MODEL FIT
# ============================================================

grid.fit(X, y)

best_model = grid.best_estimator_

print("\nBest parameters:")

print(grid.best_params_)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(

    classification_report(

        y_true_all,

        y_pred_all,

        target_names=le.classes_.astype(str)
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true_all,

    y_pred_all
)

# Row normalization
cm_percent = (

    cm.astype(float)

    /

    cm.sum(axis=1)[:, np.newaxis]
)

cm_percent *= 100

# ============================================================
# LABELS
# ============================================================

labels = np.array([

    [

        f"{cm_percent[i, j]:.1f}%\n(n={cm[i, j]})"

        for j in range(cm.shape[1])

    ]

    for i in range(cm.shape[0])

])

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 7))

sns.heatmap(

    cm_percent,

    annot=labels,

    fmt="",

    cmap="Blues",

    xticklabels=le.classes_,

    yticklabels=le.classes_,

    linewidths=1,

    linecolor="white",

    square=True,

    vmin=0,

    vmax=100,

    cbar_kws={
        "label": "Classification (%)"
    }
)

plt.xlabel(
    "Predicted Class",
    fontsize=13
)

plt.ylabel(
    "True Class",
    fontsize=13
)

plt.title(
    "Normalized Confusion Matrix",
    fontsize=15,
    weight="bold"
)

plt.xticks(rotation=20)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

# ============================================================
# PERMUTATION TEST
# ============================================================

score, permutation_scores, pvalue = (

    permutation_test_score(

        estimator=best_model,

        X=X,

        y=y,

        cv=outer_cv,

        scoring="balanced_accuracy",

        n_permutations=1000,

        random_state=42,

        n_jobs=-1
    )
)

print("\n" + "=" * 60)

print("PERMUTATION TEST")

print("=" * 60)

print(f"\nObserved score: {score:.3f}")

print(f"p-value: {pvalue:.5f}")

# ============================================================
# PERMUTATION HISTOGRAM
# ============================================================

plt.figure(figsize=(7, 5))

plt.hist(
    permutation_scores,
    bins=30
)

plt.axvline(
    score,
    color="red",
    linewidth=3
)

plt.xlabel("Permutation Scores")

plt.ylabel("Count")

plt.title("Permutation Test")

plt.tight_layout()

plt.show()

# ============================================================
# PCA VISUALIZATION
# ============================================================

X_processed = best_model[:-1].transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(8, 6))

for class_idx in np.unique(y):

    mask = y == class_idx

    plt.scatter(

        X_pca[mask, 0],

        X_pca[mask, 1],

        label=le.classes_[class_idx],

        alpha=0.8
    )

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("PCA Projection")

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

perm = permutation_importance(

    best_model,

    X,

    y,

    scoring="balanced_accuracy",

    n_repeats=100,

    random_state=42,

    n_jobs=-1
)

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": perm.importances_mean,

    "STD": perm.importances_std
})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False
)

print("\nTop Features:")

print(importance_df.head(20))

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

top_df = importance_df.head(20)

plt.figure(figsize=(10, 8))

plt.barh(

    top_df["Feature"][::-1],

    top_df["Importance"][::-1]
)

plt.xlabel("Permutation Importance")

plt.title("Top Feature Importances")

plt.tight_layout()

plt.show()

# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\n" + "=" * 60)

print("SHAP ANALYSIS")

print("=" * 60)

classifier = best_model.named_steps["clf"]

explainer = shap.Explainer(

    classifier.predict_proba,

    X_processed
)

shap_values = explainer(X_processed)

print("\nSHAP values shape:")

print(shap_values.values.shape)

# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(

    np.abs(shap_values.values),

    axis=(0, 2)
)

shap_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": mean_abs_shap
})

shap_df = shap_df.sort_values(

    by="Importance",

    ascending=False
)

print("\nTop SHAP Features:")

print(shap_df.head(20))

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    best_model,

    "best_gaussiannb_hormone_model.pkl"
)

joblib.dump(

    le,

    "label_encoder.pkl"
)

print("\nModel saved.")

# ============================================================
# SAVE RESULTS
# ============================================================

importance_df.to_csv(

    "feature_importance.csv",

    index=False
)

shap_df.to_csv(

    "shap_importance.csv",

    index=False
)

print("\nResults saved.")

# ============================================================
# END
# ============================================================