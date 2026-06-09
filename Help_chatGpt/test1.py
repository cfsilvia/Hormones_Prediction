"""
===========================================================
SEX-SPECIFIC HORMONE → ARCHETYPE PREDICTION
WITH:
- permutation testing
- SHAP analysis
- male/female separation
- LOOCV validation

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

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

import shap

import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

FILE = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Merged_Hormone_Behavior_Data.xlsx"


N_PERMUTATIONS = 300

N_PCA_COMPONENTS = 5

RANDOM_STATE = 42

# ============================================================
# LOAD
# ============================================================

df = pd.read_excel(FILE)

print(df.shape)

# ============================================================
# FIND SEX COLUMN
# ============================================================

sex_col = None

for c in df.columns:

    if c.lower() in ["sex", "gender"]:

        sex_col = c

        break

if sex_col is None:

    raise ValueError(
        "No sex column found."
    )

print("\nSex column:", sex_col)

# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def run_analysis(
    sub_df,
    sex_name
):

    print("\n================================================")
    print(sex_name)
    print("================================================")

    # ========================================================
    # NUMERIC ONLY
    # ========================================================

    num = sub_df.select_dtypes(
        include=np.number
    )

    # ========================================================
    # TARGETS
    # ========================================================

    target_cols = [

        c for c in num.columns

        if "Probability_A" in c
    ]

    print("\nTarget columns:")
    print(target_cols)

    Y_prob = num[target_cols]

    # HARD LABELS
    y = np.argmax(
        Y_prob.values,
        axis=1
    )

    # ========================================================
    # PREDICTORS
    # ========================================================

    X_df = num.drop(
        columns=target_cols
    )

    # ========================================================
    # REMOVE LEAKAGE
    # ========================================================

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

        if any(
            k in cl
            for k in remove_keywords
        ):

            drop_cols.append(c)

        if "animal" in cl:

            drop_cols.append(c)

        if "mouse" in cl:

            drop_cols.append(c)

    X_df = X_df.drop(
        columns=list(set(drop_cols)),
        errors="ignore"
    )

    print("\nRemoved columns:")
    print(drop_cols)

    print("\nRemaining predictors:")
    print(X_df.columns.tolist())

    # ========================================================
    # PREPROCESS
    # ========================================================

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(X_df)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # ========================================================
    # PCA
    # ========================================================

    n_components = min(

        N_PCA_COMPONENTS,

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

    # ========================================================
    # CLASSIFIER
    # ========================================================

    loo = LeaveOneOut()

    pred = np.zeros_like(y)

    # save probabilities
    pred_prob = np.zeros(
        (
            len(y),
            len(np.unique(y))
        )
    )

    # ========================================================
    # LOOCV
    # ========================================================

    for train_idx, test_idx in loo.split(X_pca):

        X_train = X_pca[train_idx]
        X_test = X_pca[test_idx]

        y_train = y[train_idx]

        model = RandomForestClassifier(

            n_estimators=200,

            class_weight="balanced",

            random_state=RANDOM_STATE
        )

        model.fit(
            X_train,
            y_train
        )

        pred[test_idx] = model.predict(
            X_test
        )

        pred_prob[test_idx] = model.predict_proba(
            X_test
        )

    # ========================================================
    # METRICS
    # ========================================================

    bal_acc = balanced_accuracy_score(
        y,
        pred
    )

    print("\nBalanced accuracy:")
    print(round(bal_acc, 3))

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y,
        pred
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")

    print(
        classification_report(
            y,
            pred
        )
    )

    # ========================================================
    # PERMUTATION TEST
    # ========================================================

    print("\nRunning permutation test...")

    perm_scores = []

    for perm in range(N_PERMUTATIONS):

        y_perm = np.random.permutation(y)

        perm_pred = np.zeros_like(y_perm)

        for train_idx, test_idx in loo.split(X_pca):

            model = RandomForestClassifier(

                n_estimators=200,

                class_weight="balanced",

                random_state=RANDOM_STATE
            )

            model.fit(
                X_pca[train_idx],
                y_perm[train_idx]
            )

            perm_pred[test_idx] = model.predict(
                X_pca[test_idx]
            )

        perm_acc = balanced_accuracy_score(
            y_perm,
            perm_pred
        )

        perm_scores.append(perm_acc)

    perm_scores = np.array(perm_scores)

    p_value = np.mean(
        perm_scores >= bal_acc
    )

    print("\nPermutation p-value:")
    print(p_value)

    # ========================================================
    # PLOT PERMUTATION
    # ========================================================

    plt.figure(figsize=(6,5))

    plt.hist(
        perm_scores,
        bins=30
    )

    plt.axvline(
        bal_acc,
        color="red",
        linewidth=3,
        label=f"True = {bal_acc:.3f}"
    )

    plt.xlabel("Balanced accuracy")

    plt.ylabel("Count")

    plt.title(
        f"{sex_name} permutation test"
    )

    plt.legend()

    plt.show()

    # ========================================================
    # FINAL MODEL FOR SHAP
    # ========================================================

    final_model = RandomForestClassifier(

        n_estimators=200,

        class_weight="balanced",

        random_state=RANDOM_STATE
    )

    final_model.fit(
        X_pca,
        y
    )

    # ========================================================
    # SHAP
    # ========================================================

    print("\nRunning SHAP...")

    explainer = shap.TreeExplainer(
        final_model
    )

    shap_values = explainer.shap_values(
        X_pca
    )

    # ========================================================
    # SHAP SUMMARY
    # ========================================================

    feature_names = [

        f"PC{i+1}"

        for i in range(
            X_pca.shape[1]
        )
    ]

    shap.summary_plot(

        shap_values,

        X_pca,

        feature_names=feature_names,

        show=True
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "accuracy": bal_acc,

        "p_value": p_value,

        "confusion_matrix": cm,

        "perm_scores": perm_scores
    }

# ============================================================
# FEMALES
# ============================================================

female_df = df[
    df[sex_col].astype(str).str.lower().isin(
        ["f", "female"]
    )
]

female_results = run_analysis(
    female_df,
    "FEMALES"
)

# ============================================================
# MALES
# ============================================================

male_df = df[
    df[sex_col].astype(str).str.lower().isin(
        ["m", "male"]
    )
]

male_results = run_analysis(
    male_df,
    "MALES"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================================================")
print("FINAL SUMMARY")
print("================================================")

print("\nFemales")
print(
    "Balanced accuracy:",
    round(
        female_results["accuracy"],
        3
    )
)

print(
    "Permutation p:",
    female_results["p_value"]
)

print("\nMales")
print(
    "Balanced accuracy:",
    round(
        male_results["accuracy"],
        3
    )
)

print(
    "Permutation p:",
    male_results["p_value"]
)