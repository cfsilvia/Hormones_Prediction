import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# input: hormones_arch (DataFrame) — hormones_with_archetypes data
# output: prediction_df (DataFrame), hormone_cols (list), prediction_full_df (DataFrame)
def create_archetype_prediction_data(hormones_arch, n_per_label=20, save_path=None, full_save_path=None):
    arch_cols = [c for c in hormones_arch.columns if c.startswith('Archetype') and c.endswith('_prob')]
    if not arch_cols:
        raise ValueError('No archetype probability columns found')

    exclude_cols = {
        'Experiment', 'sex', 'Sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips',
        'Mice chips', 'Animal', 'Dominant_archetype', 'PC1', 'PC2', 'PC3'
    }
    exclude_cols.update(arch_cols)
    hormone_cols = [
        c for c in hormones_arch.columns
        if c not in exclude_cols
        and not c.startswith('cum_')
        and pd.api.types.is_numeric_dtype(hormones_arch[c])
    ]
    if not hormone_cols:
        raise ValueError('No numeric hormone feature columns found')

    datasets = []
    for arch_col in arch_cols:
        arch_df = hormones_arch.dropna(subset=[arch_col]).sort_values(arch_col, ascending=False)
        if len(arch_df) < n_per_label * 2:
            raise ValueError(
                f'{arch_col} has {len(arch_df)} rows; need at least {n_per_label * 2} '
                f'to select {n_per_label} high and {n_per_label} low probability rows'
            )

        high_df = arch_df.head(n_per_label).copy()
        low_df = arch_df.tail(n_per_label).copy()

        high_df['label'] = 1
        low_df['label'] = 0
        high_df['Prediction_archetype'] = arch_col.replace('_prob', '')
        low_df['Prediction_archetype'] = arch_col.replace('_prob', '')

        datasets.append(pd.concat([high_df, low_df], ignore_index=True))

    prediction_full_df = pd.concat(datasets, ignore_index=True)
    metadata_cols = [
        c for c in [
            'Experiment', 'sex', 'Sex', 'Type', 'Genotype', 'Hierarchy',
            'Mice.chips', 'Mice chips', 'Animal', 'Dominant_archetype'
        ]
        if c in prediction_full_df.columns
    ]
    full_cols = ['Prediction_archetype', 'label'] + metadata_cols + arch_cols + hormone_cols
    prediction_full_df = prediction_full_df[full_cols]
    prediction_df = prediction_full_df[['Prediction_archetype', 'label'] + hormone_cols]

    if save_path is not None:
        prediction_df.to_excel(save_path, index=False)
        print(f'  Saved prediction data: {save_path}')

    if full_save_path is not None:
        prediction_full_df.to_excel(full_save_path, index=False)
        print(f'  Saved full prediction data: {full_save_path}')

    return prediction_df, hormone_cols, prediction_full_df


# input: prediction_df (DataFrame), hormone_cols (list), models (list of str)
# output: results (dict) — per archetype and model predictions with LOOCV accuracy
def predict_archetype_labels_loocv(prediction_df, hormone_cols, models):
    model_objects = _get_prediction_models(models)
    results = {}

    for archetype_name, archetype_df in prediction_df.groupby('Prediction_archetype'):
        X = archetype_df[hormone_cols].apply(pd.to_numeric, errors='coerce').values
        y = archetype_df['label'].astype(int).values
        loo = LeaveOneOut()
        results[archetype_name] = {}

        for model_name, model in model_objects.items():
            predictions = np.empty(len(y), dtype=int)

            for train_idx, test_idx in loo.split(X):
                model_clone = clone(model)
                model_clone.fit(X[train_idx], y[train_idx])
                predictions[test_idx[0]] = model_clone.predict(X[test_idx])[0]

            results[archetype_name][model_name] = {
                'predictions': predictions,
                'true_labels': y,
                'accuracy': accuracy_score(y, predictions),
            }

    return results


# input: prediction_results (dict) — output from predict_archetype_labels_loocv
# output: metrics_df (DataFrame) — metrics per archetype and model
def calculate_prediction_metrics(prediction_results, save_path=None):
    rows = []

    for archetype_name, model_results in prediction_results.items():
        for model_name, result in model_results.items():
            y_true = result['true_labels']
            y_pred = result['predictions']
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            rows.append({
                'Prediction_archetype': archetype_name,
                'model': model_name,
                'accuracy': accuracy_score(y_true, y_pred),
                'fscore': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': cm.tolist(),
                'true_0_pred_0': cm[0, 0],
                'true_0_pred_1': cm[0, 1],
                'true_1_pred_0': cm[1, 0],
                'true_1_pred_1': cm[1, 1],
            })

    metrics_df = pd.DataFrame(rows)

    if save_path is not None:
        metrics_df.to_excel(save_path, index=False)
        print(f'  Saved prediction metrics: {save_path}')

    return metrics_df


# input: metrics_df (DataFrame), models (list), save_path (str)
# output: None — saves one figure with archetypes as rows and models as columns
def plot_confusion_matrices_by_archetype(metrics_df, models, save_path):
    archetypes = list(metrics_df['Prediction_archetype'].drop_duplicates())
    n_rows = len(archetypes)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for row_idx, archetype_name in enumerate(archetypes):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            match = metrics_df[
                (metrics_df['Prediction_archetype'] == archetype_name)
                & (metrics_df['model'] == model_name)
            ]

            if match.empty:
                ax.axis('off')
                continue

            row = match.iloc[0]
            cm = np.array([
                [row['true_0_pred_0'], row['true_0_pred_1']],
                [row['true_1_pred_0'], row['true_1_pred_1']],
            ])

            ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred 0', 'Pred 1'])
            ax.set_yticklabels(['True 0', 'True 1'])
            ax.set_title(f'{model_name}\nAcc={row["accuracy"]:.2f}, F={row["fscore"]:.2f}')

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')

            if col_idx == 0:
                ax.set_ylabel(archetype_name)

    fig.suptitle('Confusion Matrices by Archetype and Model', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved confusion matrices plot: {save_path}')


# input: models (list of str)
# output: model_objects (dict) — sklearn-compatible classifiers by requested name
def _get_prediction_models(models):
    available_models = {
        'SVC_linear': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='linear', class_weight='balanced')),
        ]),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0),
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ]),
        'decision_tree': DecisionTreeClassifier(class_weight='balanced', random_state=0),
        'k_neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5)),
        ]),
        'xgboost': XGBClassifier(
            n_estimators=100,
            random_state=0,
            eval_metric='logloss',
            objective='binary:logistic',
        ),
    }

    unknown_models = [model_name for model_name in models if model_name not in available_models]
    if unknown_models:
        raise ValueError(f'Unknown models requested: {unknown_models}')

    return {model_name: available_models[model_name] for model_name in models}
