# ============================================================
# HORMONE CLASSIFICATION PIPELINE
# ============================================================

# Features:
# - Logistic Regression + SMOTE
# - Repeated Stratified Cross Validation
# - Macro F1 evaluation
# - Confusion matrix
# - Permutation significance test
# - Permutation feature importance
# - SHAP explainability
# - Stable evaluation for imbalanced datasets
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib

# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    GridSearchCV,
    permutation_test_score
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.inspection import permutation_importance

# ============================================================
# IMBALANCED LEARN
# ============================================================
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE

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

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabel distribution:")
print(pd.Series(y).value_counts())

# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([

    # Missing values
    ("imputer", SimpleImputer(strategy="median")),

    # # SMOTE balancing
    # ("smote", SMOTE(
    #     random_state=42,
    #     k_neighbors=1
    # )),

    # Scaling
    ("scaler", StandardScaler()),

    # Classifier
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1
    ))
])

# ============================================================
# REPEATED STRATIFIED CV
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

# ============================================================
# CROSS VALIDATION SCORES
# ============================================================

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("\nMacro F1 scores:")
print(scores)

mean_f1 = scores.mean()
std_f1 = scores.std()

print("\nMean Macro F1:")
print(mean_f1)

print("\nSTD Macro F1:")
print(std_f1)

print(f"\nMacro F1 = {mean_f1:.3f} ± {std_f1:.3f}")

# ============================================================
# MANUAL REPEATED CV PREDICTIONS
# ============================================================

y_true_all = []
y_pred_all = []

for train_idx, test_idx in cv.split(X, y):

    # Split
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Store
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

# ============================================================
# ARRAYS
# ============================================================

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(
    y_true_all,
    y_pred_all
)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(
    y_true_all,
    y_pred_all
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

# ============================================================
# FIT FINAL MODEL ON ALL DATA
# ============================================================

pipeline.fit(X, y)

# ============================================================
# PERMUTATION SIGNIFICANCE TEST
# ============================================================

score, permutation_scores, pvalue = permutation_test_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_permutations=100,
    random_state=42,
    n_jobs=-1
)

print("\nPermutation Test")

print("Real score:")
print(score)

print("p-value:")
print(pvalue)

# ============================================================
# PERMUTATION HISTOGRAM
# ============================================================

plt.figure(figsize=(6, 4))

plt.hist(
    permutation_scores,
    bins=20
)

plt.axvline(
    score,
    color="red",
    linewidth=3
)

plt.xlabel("Permutation F1")
plt.ylabel("Count")

plt.title("Permutation Test")

plt.tight_layout()
plt.show()

# # ============================================================
# # CROSS-VALIDATED PERMUTATION IMPORTANCE
# # ============================================================

# feature_importances = []

# for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

#     print(f"\nPermutation fold {fold+1}")

#     # Split
#     X_train = X.iloc[train_idx]
#     X_test = X.iloc[test_idx]

#     y_train = y[train_idx]
#     y_test = y[test_idx]

#     # Fit ONLY on training fold
#     pipeline.fit(X_train, y_train)

#     # Permutation importance ONLY on test fold
#     perm = permutation_importance(
#         pipeline,
#         X_test,
#         y_test,
#         n_repeats=30,
#         random_state=42,
#         scoring="f1_macro"
#     )

#     feature_importances.append(
#         perm.importances_mean
#     )

# # ============================================================
# # MEAN IMPORTANCE ACROSS FOLDS
# # ============================================================

# feature_importances = np.array(feature_importances)

# mean_importance = feature_importances.mean(axis=0)

# std_importance = feature_importances.std(axis=0)

# importance_df = pd.DataFrame({

#     "feature": X.columns,

#     "mean_importance": mean_importance,

#     "std_importance": std_importance
# })

# importance_df = importance_df.sort_values(
#     by="mean_importance",
#     ascending=False
# )

# print("\nTop Features:")
# print(importance_df.head(20))
# ============================================================
# SHAP PREPARATION
# ============================================================

X_processed = pipeline[:-1].transform(X) #preprocessed without the last step

classifier = pipeline.named_steps["clf"] #extract the classifier

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    classifier,
    X_processed
)

shap_values = explainer.shap_values(X_processed)
# ============================================================
# MULTICLASS HANDLING
# ============================================================

# shap_values shape:
# [n_classes, n_samples, n_features]

shap_values = np.array(shap_values)

print("SHAP shape:")
print(shap_values.shape)
# ============================================================
# SHAP Global BAR PLOT
# ============================================================
# Mean absolute SHAP across classes
plt.figure()

mean_shap = np.mean(
    np.abs(shap_values),
    axis=2
)

shap.summary_plot(
    mean_shap,
    X_processed,
    feature_names=X.columns,
    plot_type="bar",
     max_display=X.shape[1],   # SHOW ALL FEATURES
     show =False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()
plt.show()
# ============================================================
# GLOBAL FEATURE ORDER
# ============================================================

# Mean absolute SHAP across classes
mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=2
)

# Global importance per feature
global_importance = mean_abs_shap.mean(axis=0)

# Sort descending
sorted_idx = np.argsort(global_importance)[::-1]

# Reorder feature names
ordered_feature_names = X.columns[sorted_idx]

# Reorder processed data
X_processed_sorted = X_processed[:, sorted_idx]
# ============================================================
# 4 SHAP VIOLIN PLOTS IN SAME PAGE
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(18, 12)
)

axes = axes.flatten()

for class_idx in range(4):

    plt.sca(axes[class_idx])

    # Reorder SHAP values
    shap_class_sorted = shap_values[:, sorted_idx, class_idx]

    shap.summary_plot(
        shap_class_sorted,
        X_processed_sorted,
        feature_names=ordered_feature_names,
        plot_type="violin",
        max_display=X.shape[1],
        sort=False,   # IMPORTANT
        show=False
    )

    axes[class_idx].set_title(
        f"Class {class_idx}"
    )


plt.tight_layout()

plt.show()
# ============================================================
# CLASS-SPECIFIC FEATURE IMPORTANCE
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(20, 16)
)

axes = axes.flatten()

for class_idx in range(4):

    # Mean absolute SHAP for this class
    class_importance = np.mean(
        np.abs(shap_values[:, :, class_idx]),
        axis=0
    )

    # Reorder according to GLOBAL ordering
    class_importance_sorted = class_importance[sorted_idx]

    axes[class_idx].barh(
        ordered_feature_names[::-1],
        class_importance_sorted[::-1]
    )

    axes[class_idx].set_title(
        f"Class {class_idx} Feature Importance"
    )

    axes[class_idx].set_xlabel(
        "Mean |SHAP value|"
    )

plt.tight_layout()

plt.show()
# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "best_hormone_model.pkl"
)

print("\nModel saved.")

# ============================================================
# OPTIONAL:
# HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "clf__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1
)

grid.fit(X, y)

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV score:")
print(grid.best_score_)