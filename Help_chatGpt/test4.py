# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
# LOAD DATA
# ============================================================

FILE = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Merged_Hormone_Behavior_Data.xlsx"

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

# impute missing
imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X_df)

# scale
scaler = StandardScaler()

X = scaler.fit_transform(X)

# ============================================================
# CLASSIFIER
# ============================================================

loo = LeaveOneOut()

pred = np.zeros_like(y)

# ============================================================
# LEAVE-ONE-OUT CROSS VALIDATION
# ============================================================

for train_idx, test_idx in loo.split(X):

    X_train = X[train_idx]
    X_test = X[test_idx]

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
# FINAL MODEL FOR SHAP
# ============================================================

final_model = LogisticRegression(

    max_iter=5000,

    class_weight="balanced"
)

final_model.fit(
    X,
    y
)

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    final_model,
    X
)

shap_values = explainer(
    X
)

feature_names = X_df.columns.tolist()

# ============================================================
# GLOBAL SHAP
# ============================================================

print("\n================================================")
print("GLOBAL SHAP")
print("================================================")

shap.summary_plot(

    shap_values,

    X,

    feature_names=feature_names,

    show=True
)

shap.plots.bar(
    shap_values
)

# ============================================================
# FEMALES SHAP
# ============================================================

female_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["f", "female"])
)

X_female = X[female_mask]

shap_female = explainer(
    X_female
)

print("\n================================================")
print("FEMALE SHAP")
print("================================================")

shap.summary_plot(

    shap_female,

    X_female,

    feature_names=feature_names,

    show=True
)

shap.plots.bar(
    shap_female
)

# ============================================================
# MALES SHAP
# ============================================================

male_mask = (

    df[sex_col]
    .astype(str)
    .str.lower()
    .isin(["m", "male"])
)

X_male = X[male_mask]

shap_male = explainer(
    X_male
)

print("\n================================================")
print("MALE SHAP")
print("================================================")

shap.summary_plot(

    shap_male,

    X_male,

    feature_names=feature_names,

    show=True
)

shap.plots.bar(
    shap_male
)

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
    "SHAP_Global.xlsx",
    index=False
)

shap_female_df = pd.DataFrame(
    shap_female.values.reshape(
        shap_female.values.shape[0],
        -1
    )
)

shap_female_df.to_excel(
    "SHAP_Female.xlsx",
    index=False
)

shap_male_df = pd.DataFrame(
    shap_male.values.reshape(
        shap_male.values.shape[0],
        -1
    )
)

shap_male_df.to_excel(
    "SHAP_Male.xlsx",
    index=False
)

print(
    "\nSaved all SHAP results."
)