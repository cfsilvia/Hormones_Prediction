# ============================================================
# ROBUST 3-LABEL HORMONE CLASSIFICATION PIPELINE
# ============================================================

# FEATURES:
#
# - Nested Cross Validation
# - Repeated Stratified CV
# - Hyperparameter tuning
# - Balanced Accuracy + Macro F1
# - Confidence Intervals
# - Publication-quality confusion matrix (%)
# - Permutation significance test
# - Permutation feature importance
# - PCA visualization
# - SHAP explainability
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

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
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

    # Missing values
    ("imputer", SimpleImputer(strategy="median")),

    # Scaling
    ("scaler", StandardScaler()),

    # Feature selection
    ("feature_selection", SelectKBest(
        score_func=f_classif
    )),

    # Classifier
    ("clf", GaussianNB())
])

# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "feature_selection__k": [
        5,
        10,
        15,
        20,
        "all"
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

    n_jobs=-1
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

    # Fit nested model
    grid.fit(X_train, y_train)

    best_fold_model = grid.best_estimator_

    # Predict
    y_pred = best_fold_model.predict(X_test)

    # Store
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
# PUBLICATION-QUALITY CONFUSION MATRIX (%)
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all
)

# Row normalization
cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Convert to percentage
cm_percent = cm_percent * 100

# Labels
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

plt.xticks(rotation=20)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

# ============================================================
# PERMUTATION TEST
# ============================================================

score, permutation_scores, pvalue = permutation_test_score(

    estimator=best_model,

    X=X,

    y=y,

    cv=outer_cv,

    scoring="balanced_accuracy",

    n_permutations=1000,

    random_state=42,

    n_jobs=-1
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

plt.title("PCA Projection of Hormone Profiles")

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

X_selected = best_model[:-1].transform(X)

classifier = best_model.named_steps["clf"]

selected_features = X.columns[

    best_model.named_steps[
        "feature_selection"
    ].get_support()
]

explainer = shap.Explainer(

    classifier.predict_proba,

    X_selected
)

shap_values = explainer(X_selected)

print("\nSHAP shape:")

print(shap_values.values.shape)

# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(

    np.abs(shap_values.values),

    axis=(0, 2)
)

shap_df = pd.DataFrame({

    "Feature": selected_features,

    "Importance": mean_abs_shap
})

shap_df = shap_df.sort_values(

    by="Importance",

    ascending=False
)

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
# BOOTSTRAP HISTOGRAM
# ============================================================

plt.figure(figsize=(7, 5))

plt.hist(
    bootstrap_scores,
    bins=30
)

plt.xlabel("Bootstrap Scores")

plt.ylabel("Count")

plt.title("Bootstrap Stability")

plt.tight_layout()

plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "best_hormone_model_nestedCV.pkl"
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
# OPTIONAL:
# PREDICT NEW SAMPLES
# ============================================================

# new_data = pd.read_csv("new_samples.csv")

# predictions = best_model.predict(new_data)

# predicted_labels = le.inverse_transform(predictions)

# print(predicted_labels)

# ============================================================
# END
# ============================================================
# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)

# ============================================================
# TRANSFORM DATA
# ============================================================

X_selected = best_model[:-1].transform(X)

# Selected feature names
selected_features = X.columns[
    best_model.named_steps[
        "feature_selection"
    ].get_support()
]

print("\nNumber of selected features:")
print(len(selected_features))

print("\nSelected features:")
print(selected_features)

# ============================================================
# EXTRACT CLASSIFIER
# ============================================================

classifier = best_model.named_steps["clf"]

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    classifier.predict_proba,
    X_selected
)

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

shap_values = explainer(X_selected)

print("\nSHAP values shape:")
print(shap_values.values.shape)

# Shape:
# [samples, features, classes]

# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values.values),
    axis=(0, 2)
)

shap_df = pd.DataFrame({

    "Feature": selected_features,

    "Importance": mean_abs_shap
})

shap_df = shap_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop SHAP Features:")
print(shap_df.head(20))

# ============================================================
# GLOBAL SHAP BARPLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.barh(

    shap_df["Feature"][::-1],

    shap_df["Importance"][::-1]
)

plt.xlabel("Mean |SHAP value|")

plt.title("Global SHAP Importance")

plt.tight_layout()

plt.show()

# ============================================================
# CLASS-SPECIFIC SHAP IMPORTANCE
# ============================================================

n_classes = len(le.classes_)

fig, axes = plt.subplots(

    1,

    n_classes,

    figsize=(7 * n_classes, 8)
)

if n_classes == 1:
    axes = [axes]

for class_idx in range(n_classes):

    class_importance = np.mean(

        np.abs(
            shap_values.values[:, :, class_idx]
        ),

        axis=0
    )

    class_df = pd.DataFrame({

        "Feature": selected_features,

        "Importance": class_importance
    })

    class_df = class_df.sort_values(

        by="Importance",

        ascending=False
    )

    axes[class_idx].barh(

        class_df["Feature"][::-1],

        class_df["Importance"][::-1]
    )

    axes[class_idx].set_title(

        f"Class: {le.classes_[class_idx]}"
    )

    axes[class_idx].set_xlabel(
        "Mean |SHAP|"
    )

plt.tight_layout()

plt.show()

# ============================================================
# SHAP SUMMARY PLOTS
# ============================================================

for class_idx in range(n_classes):

    print("\nGenerating SHAP summary plot for:")
    print(le.classes_[class_idx])

    shap.summary_plot(

        shap_values.values[:, :, class_idx],

        X_selected,

        feature_names=selected_features,

        max_display=20,

        show=True
    )

# ============================================================
# SHAP BEESWARM PLOT
# ============================================================

# Mean across classes
mean_shap = np.mean(
    shap_values.values,
    axis=2
)

shap.summary_plot(

    mean_shap,

    X_selected,

    feature_names=selected_features,

    max_display=20
)

# ============================================================
# SAVE SHAP RESULTS
# ============================================================

shap_df.to_csv(

    "shap_importance.csv",

    index=False
)

print("\nSHAP importance saved.")

# ============================================================
# OPTIONAL:
# FORCE PLOT FOR SINGLE SAMPLE
# ============================================================

sample_idx = 0

for class_idx in range(n_classes):

    shap.plots.waterfall(

        shap.Explanation(

            values=shap_values.values[
                sample_idx,
                :,
                class_idx
            ],

            base_values=shap_values.base_values[
                sample_idx,
                class_idx
            ],

            data=X_selected[sample_idx],

            feature_names=selected_features
        ),

        max_display=15
    )

# ============================================================
# OPTIONAL:
# DEPENDENCE PLOTS
# ============================================================

top_feature = shap_df.iloc[0]["Feature"]

top_feature_idx = list(selected_features).index(
    top_feature
)

for class_idx in range(n_classes):

    shap.dependence_plot(

        top_feature_idx,

        shap_values.values[:, :, class_idx],

        X_selected,

        feature_names=selected_features
    )

# ============================================================
# END SHAP
# ============================================================