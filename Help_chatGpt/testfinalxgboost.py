# ============================================================
# IMPORTS
# ============================================================

import os

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import shap

# ============================================================
# PATHS
# ============================================================

FILE = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Merged_Hormone_Behavior_Data.xlsx"

OUTPUT_DIR = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\OUTPUTS_XGBOOST"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATA
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

# fallback
if sex_col is None:

    for c in df.columns:

        vals = (
            df[c]
            .astype(str)
            .str.lower()
            .unique()
        )

        if any(
            v in ["m", "male", "f", "female"]
            for v in vals
        ):

            sex_col = c

            break

print("\nSex column:")
print(sex_col)

# ============================================================
# KEEP ONLY NUMERIC
# ============================================================

num = df.select_dtypes(include=np.number)

# ============================================================
# TARGETS
# ============================================================

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

# ============================================================
# PREPROCESS
# ============================================================

# ONLY IMPUTATION
# NO NORMALIZATION

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X_df)

# ============================================================
# PCA ON HORMONES
# ============================================================

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
# XGBOOST CLASSIFIER
# ============================================================

loo = LeaveOneOut()

pred = np.zeros_like(y)

pred_prob = np.zeros(
    (
        len(y),
        len(np.unique(y))
    )
)

# ============================================================
# LEAVE-ONE-OUT CROSS VALIDATION
# ============================================================

for train_idx, test_idx in loo.split(X_pca):

    X_train = X_pca[train_idx]
    X_test = X_pca[test_idx]

    y_train = y[train_idx]

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=len(np.unique(y)),

        n_estimators=200,

        max_depth=3,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        eval_metric="mlogloss"
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
# SAVE CONFUSION MATRIX PNG
# ============================================================

confusion_matrix_png = os.path.join(
    OUTPUT_DIR,
    "Confusion_Matrix.png"
)

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

plt.savefig(
    confusion_matrix_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE CONFUSION MATRIX EXCEL
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[f"True_A{i}" for i in range(cm.shape[0])],
    columns=[f"Pred_A{i}" for i in range(cm.shape[1])]
)

cm_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Confusion_Matrix.xlsx"
    )
)

# ============================================================
# TOP HORMONE LOADINGS
# ============================================================

feature_names = np.array([

    f"PC{i+1}"

    for i in range(
        X_pca.shape[1]
    )
])

loadings = pd.DataFrame(

    pca.components_.T,

    index=X_df.columns,

    columns=feature_names
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
# FINAL MODEL FOR SHAP
# ============================================================

final_model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(np.unique(y)),

    n_estimators=200,

    max_depth=3,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="mlogloss"
)

final_model.fit(
    X_pca,
    y
)

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    final_model
)

shap_values = explainer.shap_values(
    X_pca
)

# ============================================================
# GLOBAL SHAP SUMMARY
# ============================================================

print("\n================================================")
print("GLOBAL SHAP")
print("================================================")

for i in range(len(shap_values)):

    shap.summary_plot(

        shap_values[i],

        X_pca,

        feature_names=feature_names,

        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"SHAP_Summary_Archetype_{i}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# GLOBAL SHAP BARPLOTS
# ============================================================

for i in range(len(shap_values)):

    shap.summary_plot(

        shap_values[i],

        X_pca,

        feature_names=feature_names,

        plot_type="bar",

        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"SHAP_Bar_Archetype_{i}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# FEMALE SHAP
# ============================================================

female_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["f", "female"])
)

X_female = X_pca[female_mask]

for i in range(len(shap_values)):

    shap.summary_plot(

        shap_values[i][female_mask],

        X_female,

        feature_names=feature_names,

        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"SHAP_Female_Archetype_{i}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# MALE SHAP
# ============================================================

male_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["m", "male"])
)

X_male = X_pca[male_mask]

for i in range(len(shap_values)):

    shap.summary_plot(

        shap_values[i][male_mask],

        X_male,

        feature_names=feature_names,

        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"SHAP_Male_Archetype_{i}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# SAVE SHAP VALUES
# ============================================================

for i in range(len(shap_values)):

    pd.DataFrame(
        shap_values[i]
    ).to_excel(

        os.path.join(
            OUTPUT_DIR,
            f"SHAP_Archetype_{i}.xlsx"
        ),

        index=False
    )

# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": final_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "XGBoost_Feature_Importance.xlsx"
    ),
    index=False
)

print("\n================================================")
print("TOP FEATURES")
print("================================================")

print(
    importance.head(10)
)

# ============================================================
# PLOT IMPORTANCE
# ============================================================

plt.figure(figsize=(7,5))

importance.head(10).sort_values(
    "Importance"
).plot.barh(
    x="Feature",
    y="Importance",
    legend=False
)

plt.xlabel("Importance")

plt.title(
    "Top XGBoost Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Top_XGBoost_Features.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE MAIN RESULTS
# ============================================================

results = pd.DataFrame({

    "True_Label": y,

    "Predicted_Label": pred
})

results.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Hormone_Archetype_Prediction.xlsx"
    ),
    index=False
)

# ============================================================
# SAVE PREDICTION PROBABILITIES
# ============================================================

prob_df = pd.DataFrame(
    pred_prob,
    columns=[
        f"Predicted_Prob_A{i}"
        for i in range(
            pred_prob.shape[1]
        )
    ]
)

prob_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Predicted_Probabilities.xlsx"
    ),
    index=False
)

# ============================================================
# SAVE PCA LOADINGS
# ============================================================

loadings.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Hormone_PCA_Loadings.xlsx"
    )
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n================================================")
print("ALL FILES SAVED")
print("================================================")

print(OUTPUT_DIR)