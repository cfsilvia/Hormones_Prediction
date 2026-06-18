from xml.parsers.expat import model
from sklearn.metrics import make_scorer
from sklearn.model_selection import permutation_test_score
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import os
from alignment.label_alignment import compute_aggregate_confusion
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import shap


def setup_figure(n_models=4):
    fig = plt.figure(figsize=(26, 14))
    gs = GridSpec(2, n_models + 1, figure=fig, height_ratios=[1, 1])
    ax_tri = fig.add_subplot(gs[:, 0])
    model_names = ['LogReg', 'SVM', 'MLP', 'XGBoost'][:n_models]
    cm_axes = {}
    perm_axes = {}
    shap_imp_axes = {}
    for i, name in enumerate(model_names):
        cm_axes[name] = fig.add_subplot(gs[0, i + 1])
        perm_axes[name] = fig.add_subplot(gs[1, i + 1])
       
    return fig, ax_tri, cm_axes, perm_axes


def draw_triangle(ax, all_pca_coords, archetypes, counts, varexlp, iteration, attempts,
                  max_iterations=50, mapping=None):
    ax.clear()
    ax.scatter(all_pca_coords[:, 0], all_pca_coords[:, 1], alpha=0.4, s=20, c='steelblue', label='Every day')

    tri_x = [archetypes[0, 0], archetypes[1, 0], archetypes[2, 0], archetypes[0, 0]]
    tri_y = [archetypes[0, 1], archetypes[1, 1], archetypes[2, 1], archetypes[0, 1]]
    ax.plot(tri_x, tri_y, 'k-', linewidth=2, label='Archetype triangle')
    colors = ['red', 'green', 'orange']
    for v in range(3):
        ax.scatter(archetypes[v, 0], archetypes[v, 1], c=colors[v], s=120, marker='^', zorder=5)
        label = f'{v + 1}→{mapping[v] + 1}' if mapping is not None else str(v + 1)
        ax.annotate(label, archetypes[v], textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=10, color=colors[v], fontweight='bold')
        ax.annotate(f'every: {counts[v]}', archetypes[v], textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color='red')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Iteration {iteration}/{max_iterations} - Var explained: {varexlp:.3f}   '
                 f'(attempts: {attempts})')
    ax.set_aspect('equal')
    ax.margins(0.1)
    if mapping is not None:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=c, markersize=8, label=f'Ref {mapping[v] + 1}')
            for v, c in enumerate(colors)]
        ax.legend(handles=legend_elements, fontsize=7)
    else:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_confusion_matrices(cm_axes, loocv_results, y_true):
    f1_scores = []
    per_class_f1 = []
    for mname, mres in loocv_results.items():
        ax = cm_axes[mname]
        ax.clear()
        f1_macro = f1_score(y_true, mres['predictions'], average='macro')
        f1_scores.append(f1_macro)
        f1_per = f1_score(y_true, mres['predictions'], average=None)
        per_class_f1.append(f1_per)
        ConfusionMatrixDisplay.from_predictions(
            y_true, mres['predictions'],
            display_labels=[1, 2, 3], ax=ax,
             cmap='Blues',
            colorbar=False, text_kw={'fontsize': 9},
        )
        ax.set_title(f'{mname}  acc={mres["accuracy"]:.3f}  F1={f1_macro:.3f}', fontsize=10)
    return f1_scores, per_class_f1


def _f1_from_cm(cm):
    n = cm.shape[0]
    f1_per = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1_per.append(f1)
    acc = cm.diagonal().sum() / cm.sum()
    return acc, np.mean(f1_per), f1_per


def plot_aggregate_confusion(agg_confusion, directory_path, model_names=None):
    if model_names is None:
        model_names = list(agg_confusion.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, mname in zip(axes, model_names):
        cm = agg_confusion[mname]
        ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3]).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False, text_kw={'fontsize': 9})
        acc, f1_macro, f1_per = _f1_from_cm(cm)
        subtitle = f'acc={acc:.3f}  F1={f1_macro:.3f}'
        for k, f1k in enumerate(f1_per):
            subtitle += f'  F1_{k+1}={f1k:.3f}'
        ax.set_title(f'{mname} — aggregate', fontsize=10)
        ax.text(0.5, -0.2, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
    plt.tight_layout()
    fname = 'aggregate_confusion.pdf'
    save_path = os.path.join(directory_path, fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved aggregate confusion: {fname}')
    plt.close(fig)


def permutation_test_significance(y_true, y_pred, n_permutations=1000):
    
    class _FixedPredictor(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            return self
        def predict(self, X):
            return y_pred
        def score(self, X, y):
            return f1_score(y, y_pred, average='macro')

    observed = f1_score(y_true, y_pred, average='macro')
    X_dummy = np.zeros((len(y_true), 1))
    _, null_scores, p_value = permutation_test_score(_FixedPredictor(), X_dummy, y_true,cv=[(slice(None), slice(None))],
        n_permutations=n_permutations,
        scoring=make_scorer(f1_score, average='macro'),
    )
    return observed, null_scores, p_value


def plot_permutation_tests(perm_axes, loocv_results, y_true, n_permutations=1000):
    pvalues = {}
    for mname, mres in loocv_results.items():
        ax = perm_axes[mname]
        ax.clear()
        observed, null_scores, p_val = permutation_test_significance(
            y_true, mres['predictions'], n_permutations)
        pvalues[mname] = p_val
        
        ax.hist(null_scores, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)
        ax.axvline(observed, color='red', linewidth=2, label=f'Observed: {observed:.3f}')
        ax.axvline(np.percentile(null_scores, 95), color='orange', linestyle='--', label='95th percentile')
        ax.set_xlabel('F1 macro')
        ax.set_ylabel('Density')
        ax.set_title(f'{mname}\np={p_val:.4f}', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    return pvalues

'''
 aggregate permutation test
'''
def plot_aggregate_permutation(all_true, all_preds_by_model, directory_path, n_permutations=1000):
    model_names = list(all_preds_by_model[0].keys())
    n = len(model_names)

    combined = {}
    for mname in model_names:
        yt = np.concatenate(all_true)
        yp = np.concatenate([entry[mname] for entry in all_preds_by_model])
        combined[mname] = permutation_test_significance(yt, yp, n_permutations)

    
    agg_confusion = compute_aggregate_confusion(all_true, all_preds_by_model)
    
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, mname in enumerate(model_names):
        ax_cm = axes[0, idx]
        cm = agg_confusion[mname]
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cm_norm, display_labels=[1, 2, 3]).plot(
            ax=ax_cm, cmap='Blues', values_format='.0%', colorbar=False, text_kw={'fontsize': 9})
        acc, f1_macro, f1_per = _f1_from_cm(cm)
        subtitle = f'acc={acc:.3f}  F1={f1_macro:.3f}'
        for k, f1k in enumerate(f1_per):
            subtitle += f'  F1_{k+1}={f1k:.3f}'
        ax_cm.set_title(f'{mname} — aggregate', fontsize=10)
        ax_cm.text(0.5, -0.2, subtitle, transform=ax_cm.transAxes, ha='center', fontsize=8)

        ax_perm = axes[1, idx]
        observed, null_scores, p_val = combined[mname]
        ax_perm.hist(null_scores, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)
        ax_perm.axvline(observed, color='red', linewidth=2, label=f'Observed: {observed:.3f}')
        ax_perm.axvline(np.percentile(null_scores, 95), color='orange', linestyle='--', label='95th percentile')
        ax_perm.set_xlabel('F1 macro')
        ax_perm.set_ylabel('Density')
        ax_perm.set_title(f'{mname}\np={p_val:.4f}', fontsize=10)
        ax_perm.legend(fontsize=8)
        ax_perm.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'aggregate_permutation.pdf'
    save_path = os.path.join(directory_path, fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved aggregate permutation test: {fname}')
    plt.close(fig)


def compute_shap_values(model, X, model_name):
    
        if model_name == 'XGBoost':
           explainer = shap.TreeExplainer(model)
        elif model_name in ('LogReg', 'SVM'):
            explainer = shap.LinearExplainer(model, X)
        else:
            explainer = shap.PermutationExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X)
        return shap_values


def train_full_models_and_shap(X_full, y_encoded, feature_cols, shap_imp_axes=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    full_models = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', max_iter=10000, probability=True),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(30,), alpha=0.1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
                                  eval_metric='mlogloss', objective='multi:softmax', num_class=3),
    }

    shap_results = {}
    for mname, model in full_models.items():
        if mname == 'XGBoost':
            model.fit(X_full, y_encoded)
            shap_vals = compute_shap_values(model, X_full, mname)
        else:
            model.fit(X_scaled, y_encoded)
            shap_vals = compute_shap_values(model, X_scaled, mname)
        shap_results[mname] = shap_vals
        if shap_imp_axes is not None:
            plot_shap_feature_importance(shap_imp_axes[mname], shap_vals, feature_cols, mname)

    return shap_results




def plot_shap_feature_importance(ax, shap_values, feature_names, model_name, top_n=10):
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values).transpose(1, 2, 0)
    if shap_values.ndim == 3:
        mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_shap = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_shap)[-top_n:]
    ax.barh(range(top_n), mean_shap[top_idx], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7)
    ax.set_xlabel('mean |SHAP|')
    ax.set_title(f'{model_name} — Top features', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')


def plot_shap_summary_violin(ax, shap_values, feature_names, model_name, top_n=10):
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values).transpose(1, 2, 0)
    if shap_values.ndim == 3:
        shap_2d = np.mean(np.abs(shap_values), axis=2)
    else:
        shap_2d = shap_values.copy()
    mean_shap = np.mean(np.abs(shap_2d), axis=0)
    top_idx = np.argsort(mean_shap)[-top_n:]
    data = [shap_2d[:, i] for i in reversed(top_idx)]
    parts = ax.violinplot(data, vert=False, positions=range(top_n), showmeans=True)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in reversed(top_idx)], fontsize=7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('SHAP value')
    ax.set_title(f'{model_name} — SHAP violin', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')


def plot_shap_per_archetype(aggregated_shap, feature_cols, directory_path):
    n_classes = 3
    model_names = list(aggregated_shap.keys())
    archetype_labels = [f'Archetype {i+1}' for i in range(n_classes)]

    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'ytick.labelsize': 14, 'xtick.labelsize': 14
    })
    for mname in model_names:
        mdata = aggregated_shap[mname]
        all_shap = np.concatenate(mdata['raw_shap'], axis=0)
        all_features = np.concatenate(mdata['raw_features'], axis=0)

        fig = plt.figure(figsize=(7 * n_classes, 6))
        for cls in range(n_classes):
            plt.subplot(1, n_classes, cls + 1)
            shap.summary_plot(
                all_shap[:, :, cls], all_features,
                feature_names=feature_cols,
                plot_type='bar',
                show=False
            )
            plt.title(f'{mname} — {archetype_labels[cls]}', fontsize=16)

        plt.tight_layout()
        fname = f'shap_{mname}_per_archetype.pdf'
        save_path = os.path.join(directory_path, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved SHAP per archetype: {fname}')
        plt.close(fig)


def save_if_best(fig, directory_path, f1_scores, per_class_f1, iteration, threshold=0.48):
    avg_f1_per = np.mean(per_class_f1, axis=0)
    if np.max(f1_scores) > threshold:
        mean_f1 = np.max(f1_scores)
        fname = f'best_f1_{mean_f1:.3f}_f1-1_{avg_f1_per[0]:.3f}_f1-2_{avg_f1_per[1]:.3f}_f1-3_{avg_f1_per[2]:.3f}_iter{iteration}.pdf'
        save_path = os.path.join(directory_path, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {fname}')
        return True
    return False
