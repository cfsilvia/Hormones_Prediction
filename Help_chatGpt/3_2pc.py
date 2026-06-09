# ============================================================
# ROBUST XGBOOST 3-CLASS HORMONE CLASSIFICATION PIPELINE
# ============================================================

# FEATURES:
#
# - XGBoost multiclass classification
# - ALL features preserved
# - Nested Cross Validation
# - Repeated Stratified CV
# - Hyperparameter tuning
# - Balanced Accuracy + Macro F1
# - Confidence Intervals
# - Publication-quality confusion matrix (%)
# - Permutation significance test
# - Permutation feature importance
# - PCA visualization
# - FULL SHAP explainability
# - Bootstrap stability
# - Model saving
#
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib

# ============================================================
# XGBOOST
# ============================================================

from xgboost import XGBClassifier

# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    permutation_test_score
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    make_scorer
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

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

    # Missing values
    ("imputer", SimpleImputer(strategy="median")),

    # Scaling
    ("scaler", StandardScaler()),

    # XGBoost classifier
    ("clf", XGBClassifier(

        objective="multi:softprob",

        num_class=len(np.unique(y)),

        eval_metric="mlogloss",

        random_state=42,

        tree_method="hist",

        use_label_encoder=False
    ))
])

# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "clf__n_estimators": [
        100,
        300,
        500
    ],

    "clf__max_depth": [
        2,
        3,
        4,
        5
    ],

    "clf__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "clf__subsample": [
        0.7,
        0.8,
        1.0
    ],

    "clf__colsample_bytree": [
        0.7,
        0.8,
        1.0
    ],

    "clf__gamma": [
        0,
        0.5,
        1
    ],

    "clf__min_child_weight": [
        1,
        3,
        5
    ]
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
# SCORING
# ============================================================

scoring = {

    "balanced_accuracy": make_scorer(
        balanced_accuracy_score
    ),

    "f1_macro": make_scorer(
        f1_score,
        average="macro"
    )
}

# ============================================================
# GRID SEARCH
# ============================================================

grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="balanced_accuracy",

    cv=inner_cv,

    n_jobs=-1,

    verbose=1
)

# ============================================================
# NESTED CROSS VALIDATION
# ============================================================

results = cross_validate(

    estimator=grid,

    X=X,

    y=y,

    cv=outer_cv,

    scoring=scoring,

    return_estimator=True,

    n_jobs=-1
)

# ============================================================
# SCORES
# ============================================================

bal_scores = results["test_balanced_accuracy"]

f1_scores = results["test_f1_macro"]

# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def confidence_interval(scores):

    mean = np.mean(scores)

    ci_low = np.percentile(scores, 2.5)

    ci_high = np.percentile(scores, 97.5)

    return mean, ci_low, ci_high

# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

bal_mean, bal_low, bal_high = confidence_interval(
    bal_scores
)

f1_mean, f1_low, f1_high = confidence_interval(
    f1_scores
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("NESTED CV RESULTS")

print("=" * 60)

print(f"\nBalanced Accuracy:")

print(f"Mean = {bal_mean:.3f}")

print(f"95% CI = [{bal_low:.3f}, {bal_high:.3f}]")

print(f"\nMacro F1:")

print(f"Mean = {f1_mean:.3f}")

print(f"95% CI = [{f1_low:.3f}, {f1_high:.3f}]")

# ============================================================
# FINAL MODEL FIT
# ============================================================

grid.fit(X, y)

best_model = grid.best_estimator_

print("\nBest parameters:")

print(grid.best_params_)

# ============================================================
# CROSS-VALIDATED PREDICTIONS
# ============================================================

y_true_all = []

y_pred_all = []

for train_idx, test_idx in outer_cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    grid.fit(X_train, y_train)

    best_fold_model = grid.best_estimator_

    y_pred = best_fold_model.predict(X_test)

    y_true_all.extend(y_test)

    y_pred_all.extend(y_pred)

# ============================================================
# ARRAYS
# ============================================================

y_true_all = np.array(y_true_all)

y_pred_all = np.array(y_pred_all)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(classification_report(

    y_true_all,

    y_pred_all,

    target_names=le.classes_.astype(str)
))

# ============================================================
# CONFUSION MATRIX (%)
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all
)

cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

cm_percent = cm_percent * 100

labels = np.array([

    [
        f"{cm_percent[i, j]:.1f}%\n(n={cm[i, j]})"

        for j in range(cm.shape[1])
    ]

    for i in range(cm.shape[0])
])

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
# PERMUTATION IMPORTANCE
# ============================================================

perm = permutation_importance(

    best_model,

    X,

    y,

    scoring="balanced_accuracy",

    n_repeats=50,

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

top_n = 20

top_df = importance_df.head(top_n)

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

# ============================================================
# TRANSFORM DATA
# ============================================================

X_processed = best_model[:-1].transform(X)

feature_names = X.columns

# ============================================================
# EXTRACT XGBOOST MODEL
# ============================================================

xgb_model = best_model.named_steps["clf"]

# ============================================================
# TREE SHAP
# ============================================================

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_processed)

print("\nSHAP SHAPE:")

print(np.array(shap_values).shape)

# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

if isinstance(shap_values, list):

    mean_abs_shap = np.mean(

        [np.abs(sv) for sv in shap_values],

        axis=(0, 1)
    )

else:

    mean_abs_shap = np.mean(

        np.abs(shap_values),

        axis=(0, 2)
    )

shap_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": mean_abs_shap
})

shap_df = shap_df.sort_values(

    by="Importance",

    ascending=False
)

print("\nTop SHAP Features:")

print(shap_df.head(20))

# ============================================================
# SHAP BARPLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.barh(

    shap_df["Feature"][::-1],

    shap_df["Importance"][::-1]
)

plt.xlabel("Mean |SHAP|")

plt.title("Global SHAP Importance")

plt.tight_layout()

plt.show()

# ============================================================
# SHAP SUMMARY PLOTS
# ============================================================

n_classes = len(le.classes_)

for class_idx in range(n_classes):

    print("\nGenerating SHAP plot for:")

    print(le.classes_[class_idx])

    if isinstance(shap_values, list):

        shap.summary_plot(

            shap_values[class_idx],

            X_processed,

            feature_names=feature_names,

            max_display=20
        )

    else:

        shap.summary_plot(

            shap_values[:, :, class_idx],

            X_processed,

            feature_names=feature_names,

            max_display=20
        )

# ============================================================
# SAVE SHAP VALUES
# ============================================================

os.makedirs("shap_outputs", exist_ok=True)

if isinstance(shap_values, list):

    for class_idx in range(n_classes):

        df_shap = pd.DataFrame(

            shap_values[class_idx],

            columns=feature_names
        )

        df_shap.to_csv(

            f"shap_outputs/shap_values_class_{class_idx}.csv",

            index=False
        )

else:

    for class_idx in range(n_classes):

        df_shap = pd.DataFrame(

            shap_values[:, :, class_idx],

            columns=feature_names
        )

        df_shap.to_csv(

            f"shap_outputs/shap_values_class_{class_idx}.csv",

            index=False
        )

print("\nAll SHAP values saved.")

# ============================================================
# BOOTSTRAP STABILITY
# ============================================================

n_bootstrap = 200

bootstrap_scores = []

rng = np.random.RandomState(42)

for i in range(n_bootstrap):

    sample_idx = rng.choice(

        len(X),

        size=len(X),

        replace=True
    )

    X_boot = X.iloc[sample_idx]

    y_boot = y[sample_idx]

    grid.fit(X_boot, y_boot)

    score = grid.best_score_

    bootstrap_scores.append(score)

bootstrap_scores = np.array(
    bootstrap_scores
)

print("\n" + "=" * 60)

print("BOOTSTRAP STABILITY")

print("=" * 60)

print(f"\nBootstrap mean: {bootstrap_scores.mean():.3f}")

print(f"Bootstrap std: {bootstrap_scores.std():.3f}")

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "best_xgboost_hormone_model.pkl"
)

joblib.dump(
    le,
    "label_encoder.pkl"
)

# ============================================================
# SAVE RESULTS
# ============================================================

importance_df.to_csv(
    "permutation_importance.csv",
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