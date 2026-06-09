# ============================================================
# IMPORTS
# ============================================================

import os

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

import shap

# ============================================================
# PATHS
# ============================================================

FILE = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Merged_Hormone_Behavior_Data.xlsx"

OUTPUT_DIR = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\OUTPUTS"

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

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X_df)

scaler = StandardScaler()

X = scaler.fit_transform(X)

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

final_model = LogisticRegression(

    max_iter=5000,

    class_weight="balanced"
)

final_model.fit(
    X_pca,
    y
)

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    final_model,
    X_pca
)

shap_values = explainer(
    X_pca
)

# ============================================================
# GLOBAL SHAP SUMMARY
# ============================================================

print("\n================================================")
print("GLOBAL SHAP")
print("================================================")

for i in range(shap_values.values.shape[2]):

    shap.summary_plot(

        shap_values.values[:,:,i],

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

for i in range(shap_values.values.shape[2]):

    shap_expl = shap.Explanation(

        values=shap_values.values[:,:,i],

        base_values=shap_values.base_values[:,i],

        data=X_pca,

        feature_names=feature_names
    )

    shap.plots.bar(

        shap_expl,

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
# FEMALES SHAP
# ============================================================

female_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["f", "female"])
)

X_female = X_pca[female_mask]

shap_female = explainer(
    X_female
)

for i in range(shap_female.values.shape[2]):

    shap.summary_plot(

        shap_female.values[:,:,i],

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
# MALES SHAP
# ============================================================

male_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["m", "male"])
)

X_male = X_pca[male_mask]

shap_male = explainer(
    X_male
)

for i in range(shap_male.values.shape[2]):

    shap.summary_plot(

        shap_male.values[:,:,i],

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

shap_df = pd.DataFrame(
    shap_values.values.reshape(
        shap_values.values.shape[0],
        -1
    )
)

shap_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "SHAP_Global.xlsx"
    ),
    index=False
)

shap_female_df = pd.DataFrame(
    shap_female.values.reshape(
        shap_female.values.shape[0],
        -1
    )
)

shap_female_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "SHAP_Female.xlsx"
    ),
    index=False
)

shap_male_df = pd.DataFrame(
    shap_male.values.reshape(
        shap_male.values.shape[0],
        -1
    )
)

shap_male_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "SHAP_Male.xlsx"
    ),
    index=False
)

# ============================================================
# MAIN FEATURES DRIVING EACH ARCHETYPE
# ============================================================

print("\n================================================")
print("MAIN FEATURES PER ARCHETYPE")
print("================================================")

coef_df = pd.DataFrame(

    final_model.coef_.T,

    index=feature_names,

    columns=[

        f"Archetype_{i}"

        for i in range(
            final_model.coef_.shape[0]
        )
    ]
)

coef_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Archetype_Feature_Weights.xlsx"
    )
)

top_features_all = []

for arch in coef_df.columns:

    print("\n--------------------------------------------")
    print(arch)
    print("--------------------------------------------")

    pos = (
        coef_df[arch]
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTOP POSITIVE FEATURES")
    print(pos)

    neg = (
        coef_df[arch]
        .sort_values(ascending=True)
        .head(10)
    )

    print("\nTOP NEGATIVE FEATURES")
    print(neg)

    for feat, val in pos.items():

        top_features_all.append({

            "Archetype": arch,

            "Feature": feat,

            "Direction": "Positive",

            "Weight": val
        })

    for feat, val in neg.items():

        top_features_all.append({

            "Archetype": arch,

            "Feature": feat,

            "Direction": "Negative",

            "Weight": val
        })

top_features_df = pd.DataFrame(
    top_features_all
)

top_features_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Top_Features_Per_Archetype.xlsx"
    ),
    index=False
)

# ============================================================
# PLOT TOP FEATURES
# ============================================================

for arch in coef_df.columns:

    vals = (
        coef_df[arch]
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(7,5))

    vals.sort_values().plot.barh()

    plt.xlabel("Absolute coefficient")

    plt.title(
        f"Most important features\n{arch}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{arch}_Top_Features.png"
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

print("\n================================================")
print("ALL FILES SAVED")
print("================================================")

print(OUTPUT_DIR)