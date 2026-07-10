from xml.parsers.expat import model
from sklearn.base import clone
from archetype_analysis import _get_models
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'pdf.use14corefonts': False,
    'image.composite_image': False,
})
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from sklearn.metrics import f1_score, confusion_matrix
import os
import pandas as pd
from alignment.label_alignment import compute_aggregate_confusion
import shap
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import Normalize, to_rgba
from scipy.stats import chi2, pearsonr
from statsmodels.stats.multitest import multipletests


def _disable_rasterization(fig):
    for artist in fig.findobj():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(False)


def _draw_vector_matrix(ax, values, labels=None, cmap='Blues', vmin=None, vmax=None,
                        text_labels=None, text_format='.1%', text_size=9):
    values = np.asarray(values, dtype=float)
    n_rows, n_cols = values.shape
    cmap_obj = plt.get_cmap(cmap)
    finite_values = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            facecolor = cmap_obj(norm(value)) if np.isfinite(value) else 'white'
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                   facecolor=facecolor, edgecolor='white', linewidth=1))

            label = None
            if text_labels is not None:
                label = text_labels[row][col]
            elif np.isfinite(value):
                label = format(value, text_format)
            if label:
                text_color = 'white' if norm(value) > 0.55 else 'black'
                ax.text(col, row, label, ha='center', va='center',
                        color=text_color, fontsize=text_size)

    if labels is not None:
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ScalarMappable(norm=norm, cmap=cmap_obj)


def setup_figure(model_names=None):
    if model_names is None:
        model_names = ['LogReg', 'SVM', 'MLP', 'RandomForest', 'ExtraTrees', 'HistGB', 'XGBoost']
    n_models = len(model_names)
    fig = plt.figure(figsize=(5 * (n_models + 1), 14))
    gs = GridSpec(2, n_models + 1, figure=fig, height_ratios=[1, 1])
    ax_tri = fig.add_subplot(gs[:, 0])
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
        cm = confusion_matrix(y_true, mres['predictions'], labels=[1, 2, 3])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

        _draw_vector_matrix(ax, cm_normalized, labels=[1, 2, 3], cmap='Blues',
                            vmin=0, vmax=1, text_format='.1%', text_size=9)

        ax.set_title(f'{mname}  acc={mres["accuracy"]:.3f}  F1={f1_macro:.3f}', fontsize=10)
    return f1_scores, per_class_f1


def plot_loocv_confusion_matrices(loocv_results, y_true, save_path, labels=None, title=None, permutation_results=None):
    if labels is None:
        labels = [1, 2, 3]
    if title is None:
        title = 'LOOCV Confusion Matrix - Mean Hormones Archetype Prediction'

    n_models = len(loocv_results)
    has_permutation = permutation_results is not None
    n_rows = 2 if has_permutation else 1
    fig, axes = plt.subplots(n_rows, n_models, figsize=(5 * n_models, 4 * n_rows), squeeze=False)

    for col_idx, (model_name, result) in enumerate(loocv_results.items()):
        ax = axes[0, col_idx]
        y_pred = result['predictions']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        acc = np.mean(np.asarray(y_true) == np.asarray(y_pred))
        f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_text = ' '.join(f'F{label}={score:.3f}' for label, score in zip(labels, f1_per_class))
        p_text = ''
        if permutation_results is not None and model_name in permutation_results:
            model_perm = permutation_results[model_name]
            p_value = model_perm['p_value'] if isinstance(model_perm, dict) else model_perm
            p_text = f', p={p_value:.4f}'

        _draw_vector_matrix(ax, cm_percent, labels=labels, cmap='Blues',
                            vmin=0, vmax=1, text_format='.1%', text_size=9)
        ax.set_title(f'{model_name}\nAcc={acc:.3f}, F1={f1_macro:.3f}{p_text}\n{f1_text}')

        if has_permutation:
            perm_ax = axes[1, col_idx]
            model_perm = permutation_results.get(model_name)
            if model_perm is None:
                perm_ax.axis('off')
                continue

            null_scores = model_perm['null_scores']
            observed = model_perm['observed']
            p_value = model_perm['p_value']

            perm_ax.hist(null_scores, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)
            perm_ax.axvline(observed, color='red', linewidth=2, label=f'Observed: {observed:.3f}')
            perm_ax.axvline(np.percentile(null_scores, 95), color='orange', linestyle='--', label='95th percentile')
            perm_ax.set_xlabel('F1 macro')
            perm_ax.set_ylabel('Density')
            perm_ax.set_title(f'{model_name} permutation\np={p_value:.4f}', fontsize=10)
            perm_ax.legend(fontsize=8)
            perm_ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved LOOCV confusion matrix: {save_path}')


def calculate_loocv_permutation_pvalues(hormones_arch, metadata_cols, loocv_results, y_true,
                                        model_names=None, n_permutations=200):
    feature_cols = [c for c in hormones_arch.columns
                    if c not in metadata_cols and c != 'Dominant_archetype'
                    and c != 'PC1' and c != 'PC2'
                    and c not in ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']]
    X = hormones_arch[feature_cols].select_dtypes(include=[np.number]).values
    labels = np.unique(y_true)
    model_lookup = _get_models(n_classes=len(labels), model_names=model_names)
    permutation_results = {}

    for model_name, result in loocv_results.items():
        observed = f1_score(y_true, result['predictions'], labels=labels,
                            average='macro', zero_division=0)
        null_scores, null_per_class, p_value = permutation_test_significance(
            X, y_true, model_lookup[model_name], model_name, observed, n_permutations)
        permutation_results[model_name] = {
            'p_value': p_value,
            'observed': observed,
            'null_scores': null_scores,
            'null_per_class': null_per_class,
        }

    return permutation_results


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
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        text_labels = [[f'{cm_percent[i, j]:.1f}%' for j in range(cm_percent.shape[1])]
                       for i in range(cm_percent.shape[0])]
        _draw_vector_matrix(ax, cm_percent, labels=[1, 2, 3], cmap='Blues',
                            vmin=0, vmax=100, text_labels=text_labels, text_size=9)
        acc, f1_macro, f1_per = _f1_from_cm(cm)
        subtitle = f'acc={acc:.3f}  F1={f1_macro:.3f}'
        for k, f1k in enumerate(f1_per):
            subtitle += f'  F1_{k+1}={f1k:.3f}'
        ax.set_title(f'{mname} — aggregate (%)', fontsize=10)
        ax.text(0.5, -0.2, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
    plt.tight_layout()
    fname = 'aggregate_confusion.pdf'
    save_path = os.path.join(directory_path, fname)
    _disable_rasterization(fig)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved aggregate confusion: {fname}')
    plt.close(fig)


def plot_aggregate_confusion_mean_se(agg_cm_stats, directory_path, model_names=None):
    if model_names is None:
        model_names = list(agg_cm_stats.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, mname in zip(axes, model_names):
        mean_cm = agg_cm_stats[mname]['mean_cm']
        std_cm = agg_cm_stats[mname]['std_cm']
        n_iter = agg_cm_stats[mname]['n_iterations']

        n_classes = mean_cm.shape[0]
        text_labels = [[f'{mean_cm[i, j] * 100:.1f}% ± {std_cm[i, j] * 100:.1f}%'
                        for j in range(n_classes)] for i in range(n_classes)]
        _draw_vector_matrix(ax, mean_cm, labels=[1, 2, 3], cmap='Blues',
                            vmin=0, vmax=1, text_labels=text_labels, text_size=9)

        summed_cm = mean_cm * n_iter
        acc, f1_macro, f1_per = _f1_from_cm(summed_cm)
        subtitle = f'acc={acc:.3f}  F1={f1_macro:.3f}'
        for k, f1k in enumerate(f1_per):
            subtitle += f'  F1_{k+1}={f1k:.3f}'
        ax.set_title(f'{mname} — mean ± std (n={n_iter})', fontsize=10)
        ax.text(0.5, -0.2, subtitle, transform=ax.transAxes, ha='center', fontsize=8)
    plt.tight_layout()
    fname = 'aggregate_confusion_mean_se.pdf'
    save_path = os.path.join(directory_path, fname)
    _disable_rasterization(fig)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved mean ± std confusion: {fname}')
    plt.close(fig)


def _fit_predict_loocv_for_model(model, model_name, X, y):
    

    preds = np.empty(len(y), dtype=y.dtype)
    for train_idx, test_idx in LeaveOneOut().split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        model_clone = clone(model)

        if model_name in ('HistGB', 'XGBoost'):
            classes = np.unique(y_train)
            cw = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([cw[list(classes).index(v)] for v in y_train])
            model_clone.fit(X_train, y_train, sample_weight=sample_weights)
            preds[test_idx[0]] = model_clone.predict(X_test)[0]
        elif model_name in ('RandomForest', 'ExtraTrees'):
            model_clone.fit(X_train, y_train)
            preds[test_idx[0]] = model_clone.predict(X_test)[0]
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            if model_name == 'MLP':
                classes = np.unique(y_train)
                cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
                sample_weights = np.array([cw[v] for v in y_train])
                model_clone.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model_clone.fit(X_train_scaled, y_train)
            preds[test_idx[0]] = model_clone.predict(X_test_scaled)[0]
    return preds


def permutation_test_significance(X, y_true, model, model_name, observed, n_permutations=200):
    from sklearn.preprocessing import LabelEncoder

    y_true = np.asarray(y_true)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_true)
    labels = label_encoder.classes_

    rng = np.random.default_rng(0)
    null_scores = np.zeros(n_permutations)
    null_per_class = np.zeros((n_permutations, len(labels)))
    for i in range(n_permutations):
        y_perm = rng.permutation(y_encoded)
        perm_pred = label_encoder.inverse_transform(_fit_predict_loocv_for_model(model, model_name, X, y_perm))
        y_perm_labels = label_encoder.inverse_transform(y_perm)
        null_scores[i] = f1_score(y_perm_labels, perm_pred, average='macro')
        null_per_class[i] = f1_score(y_perm_labels, perm_pred, labels=labels, average=None, zero_division=0)

    p_value = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)
    return null_scores, null_per_class, p_value


def fixed_prediction_permutation_test(y_true, y_pred, n_permutations=1000):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observed = f1_score(y_true, y_pred, average='macro')
    rng = np.random.default_rng(0)
    null_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_true)
        null_scores[i] = f1_score(y_perm, y_pred, average='macro')
    p_value = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)
    return observed, null_scores, p_value


def plot_permutation_tests(perm_axes, loocv_results, y_true, hormones_arch, metadata_cols,
                           model_names=None, n_permutations=50):
    

    pvalues = {}
    feature_cols = [c for c in hormones_arch.columns
                    if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2']
    X = hormones_arch[feature_cols].select_dtypes(include=[np.number]).values
    model_lookup = _get_models(n_classes=3, model_names=model_names)
    for mname, mres in loocv_results.items():
        ax = perm_axes[mname]
        ax.clear()
        labels = np.unique(y_true)
        observed = f1_score(y_true, mres['predictions'], average='macro')
        observed_per_class = f1_score(y_true, mres['predictions'], labels=labels,
                                      average=None, zero_division=0)
        null_scores, null_per_class, p_val = permutation_test_significance(
            X, y_true, model_lookup[mname], mname, observed, n_permutations)
        pvalues[mname] = p_val
        
        ax.hist(null_scores, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)
        ax.axvline(observed, color='red', linewidth=2, label=f'Observed: {observed:.3f}')
        ax.axvline(np.percentile(null_scores, 95), color='orange', linestyle='--', label='95th percentile')
        ax.set_xlabel('F1 macro')
        ax.set_ylabel('Density')
        per_class_text = ' '.join(f'F1_{i + 1}={score:.3f}' for i, score in enumerate(observed_per_class))
        ax.set_title(f'{mname}\np={p_val:.4f} F1={observed:.3f}\n{per_class_text}', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    return pvalues

'''
 aggregate permutation test
'''
def plot_aggregate_permutation(all_true, all_preds_by_model, directory_path, n_permutations=1000):
    model_names = list(all_preds_by_model[0].keys())
    n = len(model_names)

    from scipy.stats import chi2
    rng = np.random.RandomState(42)

    combined = {}
    for mname in model_names:
        observed_list = []
        per_iter_pvalues = []
        nulls = []
        # Run permutation test per iteration and collect per-iteration p-values + nulls
        for yt, entry in zip(all_true, all_preds_by_model):
            obs, null_scores, p_i = fixed_prediction_permutation_test(yt, entry[mname], n_permutations)
            observed_list.append(obs)
            per_iter_pvalues.append(p_i)
            nulls.append(null_scores)

        combined_observed = np.mean(observed_list)

        # Combine per-iteration p-values with Fisher's method (protect against zero p-values)
        per_iter_pvalues = np.maximum(np.array(per_iter_pvalues), 1.0 / (n_permutations + 1))
        chi2_stat = -2.0 * np.sum(np.log(per_iter_pvalues))
        p_fisher = 1.0 - chi2.cdf(chi2_stat, 2 * len(per_iter_pvalues))

        # Build a null distribution for the mean-statistic by sampling from per-iteration nulls.
        # For each combined sample, pick a random null-score from each iteration's nulls and average them.
        null_matrix = np.vstack(nulls)  # shape: (n_iterations, n_permutations)
        n_iter = null_matrix.shape[0]
        combined_null = np.zeros(n_permutations)
        for k in range(n_permutations):
            idxs = rng.randint(0, n_permutations, size=n_iter)
            combined_null[k] = np.mean(null_matrix[np.arange(n_iter), idxs])

        # empirical p-value comparing observed mean to sampled nulls
        p_empirical = (np.sum(combined_null >= combined_observed) + 1) / (n_permutations + 1)

        # store both empirical null-based p and Fisher combined p (4-tuple)
        combined[mname] = (combined_observed, combined_null, p_empirical, p_fisher)

    
    agg_confusion = compute_aggregate_confusion(all_true, all_preds_by_model)
    
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, mname in enumerate(model_names):
        ax_cm = axes[0, idx]
        cm = agg_confusion[mname]
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        _draw_vector_matrix(ax_cm, cm_norm, labels=[1, 2, 3], cmap='Blues',
                            vmin=0, vmax=1, text_format='.0%', text_size=9)
        acc, f1_macro, f1_per = _f1_from_cm(cm)
        subtitle = f'acc={acc:.3f}  F1={f1_macro:.3f}'
        for k, f1k in enumerate(f1_per):
            subtitle += f'  F1_{k+1}={f1k:.3f}'
        ax_cm.set_title(f'{mname} — aggregate', fontsize=10)
        ax_cm.text(0.5, -0.2, subtitle, transform=ax_cm.transAxes, ha='center', fontsize=8)

        ax_perm = axes[1, idx]
        observed, null_scores, p_val, p_fisher = combined[mname]
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
    _disable_rasterization(fig)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved aggregate permutation test: {fname}')
    plt.close(fig)


# input: all_true (list of lists), all_preds_by_model (list of dicts), directory_path (str), n_permutations (int)
# output: None (saves grid of per-archetype permutation histograms, rows=model, cols=archetype)
def plot_permutation_per_archetype(all_true, all_preds_by_model, directory_path, n_permutations=1000):
    from sklearn.metrics import f1_score

    model_names = list(all_preds_by_model[0].keys())
    n_models = len(model_names)
    archetype_labels = [1, 2, 3]
    arch_colors = ['red', 'green', 'orange']

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 4 * n_models),
                             sharex='col', sharey='col')

    for m_idx, mname in enumerate(model_names):
        y_true_arr = np.concatenate(all_true)
        y_pred_arr = np.concatenate([entry[mname] for entry in all_preds_by_model])

        observed_f1 = f1_score(y_true_arr, y_pred_arr, average=None)

        null_scores = {c: [] for c in archetype_labels}
        rng = np.random.RandomState(42)
        for _ in range(n_permutations):
            perm = rng.permutation(len(y_true_arr))
            f1_vals = f1_score(y_true_arr[perm], y_pred_arr, average=None)
            for i, c in enumerate(archetype_labels):
                null_scores[c].append(f1_vals[i])

        for cls_idx, cls_label in enumerate(archetype_labels):
            ax = axes[m_idx, cls_idx] if n_models > 1 else axes[cls_idx]

            null_arr = np.array(null_scores[cls_label])
            p_val = (np.sum(null_arr >= observed_f1[cls_idx]) + 1) / (n_permutations + 1)

            ax.hist(null_arr, bins=30, alpha=0.7, color='gray',
                    edgecolor='black', density=True)
            ax.axvline(observed_f1[cls_idx], color=arch_colors[cls_idx], linewidth=2,
                       label=f'Obs F1={observed_f1[cls_idx]:.3f}')
            ax.axvline(np.percentile(null_arr, 95), color='orange',
                       linestyle='--', label='95th percentile')
            ax.legend(fontsize=7, loc='upper left')
            ax.text(0.98, 0.95, f'p={p_val:.4f}', transform=ax.transAxes, fontsize=9,
                    va='top', ha='right', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
            ax.grid(True, alpha=0.3)

            if m_idx == 0:
                ax.set_title(f'Archetype {cls_label}', fontsize=12, color=arch_colors[cls_idx])
            if m_idx == n_models - 1:
                ax.set_xlabel('F1 score')
            if cls_idx == 0:
                ax.set_ylabel('Density')

    for m_idx, mname in enumerate(model_names):
        ax = axes[m_idx, 0] if n_models > 1 else axes[0]
        ax.text(-0.22, 0.5, mname, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='center', ha='right', rotation=90)

    fig.suptitle('Permutation Test — Per-Archetype F1 Scores per Model', fontsize=14)
    plt.tight_layout()
    fname = 'permutation_per_archetype.pdf'
    save_path = os.path.join(directory_path, fname)
    _disable_rasterization(fig)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved per-archetype permutation test: {fname}')
    plt.close(fig)


def compute_shap_values(model, X, model_name):
    
        if model_name in ('RandomForest', 'ExtraTrees', 'XGBoost'):
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
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    full_models = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', max_iter=10000, probability=True),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(30,), alpha=0.1),
        'RandomForest': RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
                                  eval_metric='mlogloss', objective='multi:softmax', num_class=3),
    }

    shap_results = {}
    for mname, model in full_models.items():
        if mname in ('RandomForest', 'ExtraTrees', 'HistGB', 'XGBoost'):
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
        _disable_rasterization(fig)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved SHAP per archetype: {fname}')
        plt.close(fig)


def save_if_best(fig, directory_path, f1_scores, per_class_f1, iteration, threshold=0.48):
    avg_f1_per = np.mean(per_class_f1, axis=0)
    if np.max(f1_scores) > threshold:
        mean_f1 = np.max(f1_scores)
        fname = f'best_f1_{mean_f1:.3f}_f1-1_{avg_f1_per[0]:.3f}_f1-2_{avg_f1_per[1]:.3f}_f1-3_{avg_f1_per[2]:.3f}_iter{iteration}.pdf'
        save_path = os.path.join(directory_path, fname)
        _disable_rasterization(fig)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {fname}')
        return True
    return False


def permutation_test_archetype_difference(all_archetype_probs, target_arch=3, n_permutations=1000):
    """
    Test if the target archetype probabilities are significantly different from the other two.

    Parameters:
    - all_archetype_probs: ndarray of shape (n_samples, 3) — pooled archetype probabilities across iterations
    - target_arch: int (1, 2, or 3) — which archetype to test as different
    - n_permutations: int — number of permutations

    Returns:
    - observed_diff: float — observed difference in mean probabilities
    - null_distribution: ndarray — null distribution of differences
    - p_value: float — permutation p-value
    """
    target_idx = target_arch - 1
    other_indices = [i for i in range(3) if i != target_idx]

    target_probs = all_archetype_probs[:, target_idx]
    other_probs = np.mean(all_archetype_probs[:, other_indices], axis=1)

    observed_diff = np.mean(target_probs) - np.mean(other_probs)

    rng = np.random.RandomState(42)
    null_distribution = np.zeros(n_permutations)
    n = len(target_probs)

    for i in range(n_permutations):
        perm = rng.permutation(n)
        perm_target = target_probs[perm]
        null_distribution[i] = np.mean(perm_target) - np.mean(other_probs)

    p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)

    return observed_diff, null_distribution, p_value


def plot_permutation_archetype_difference(all_archetype_probs, target_arch=3, n_permutations=1000,
                                          save_path=None):
    """
    Plot permutation test showing if target archetype is different from others.

    Parameters:
    - all_archetype_probs: ndarray of shape (n_samples, 3) — pooled archetype probabilities
    - target_arch: int (1, 2, or 3) — which archetype to test
    - n_permutations: int — number of permutations
    - save_path: str or None — path to save figure

    Returns:
    - fig, ax, p_value
    """
    observed_diff, null_distribution, p_value = permutation_test_archetype_difference(
        all_archetype_probs, target_arch, n_permutations)

    other_archs = [i for i in [1, 2, 3] if i != target_arch]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_distribution, bins=40, alpha=0.7, color='gray', edgecolor='black', density=True,
            label='Null distribution')
    ax.axvline(observed_diff, color='red', linewidth=2.5,
               label=f'Observed diff = {observed_diff:.4f}')
    ax.axvline(-observed_diff, color='red', linewidth=2.5, linestyle='--', alpha=0.5)
    ax.axvline(np.percentile(null_distribution, 97.5), color='orange', linestyle='--',
               label='97.5th percentile')
    ax.axvline(np.percentile(null_distribution, 2.5), color='orange', linestyle='--',
               label='2.5th percentile')
    ax.set_xlabel('Difference in mean probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Permutation Test: Archetype {target_arch} vs Archetypes {other_archs}\n'
                 f'p = {p_value:.4f}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        _disable_rasterization(fig)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved permutation archetype difference plot: {save_path}')

    return fig, ax, p_value

"""
    Draws the mean archetype triangle and confidence ellipses.

    accepted_results contains mapped archetypes.
    """
def plot_mean_archetype_triangle(all_pca_coords, accepted_results, output_file,confidence=0.95):
    n_vertices = accepted_results[0]['archetypes'].shape[0]
    vertex_positions = [[] for _ in range(n_vertices)]
    # collect aligned archetypes
    for result in accepted_results:
        mapping = result["mapping"]

        aligned = np.zeros_like(result["archetypes"])

        for current_idx, ref_idx in enumerate(mapping):
            aligned[ref_idx] = result["archetypes"][current_idx]

        for v in range(n_vertices):
            vertex_positions[v].append(aligned[v])
    means = []
    covs = []

    for pts in vertex_positions:
        pts = np.asarray(pts, dtype=float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]

        if len(pts) == 0:
            means.append(np.full(all_pca_coords.shape[1], np.nan))
            covs.append(np.full((all_pca_coords.shape[1], all_pca_coords.shape[1]), np.nan))
        elif len(pts) == 1:
            means.append(pts[0])
            covs.append(np.zeros((pts.shape[1], pts.shape[1])))
        else:
            means.append(np.mean(pts, axis=0))
            covs.append(np.cov(pts.T))

    


    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(all_pca_coords[:,0],all_pca_coords[:,1], s=8, alpha=.8, color="blue")
    means = np.asarray(means)
    mean_coords_df = pd.DataFrame(means, columns=[f'PC{i + 1}' for i in range(means.shape[1])])
    mean_coords_df.insert(0, 'archetype', np.arange(1, len(means) + 1)) #add archetype column to the dataframe for clarity
    
    closed = np.vstack([means, means[0]])

    ax.plot(closed[:,0], closed[:,1], color='black', linestyle='-', lw=3)
    #compute the scale factor for the confidence ellipse based on the chi-squared distribution
    scale = np.sqrt(chi2.ppf(confidence, 2))
    colors = ['pink', 'purple',  'cyan']
    ellipse_geometry = []

    for i, (mean, cov) in enumerate(zip(means, covs)):
        color = colors[i % len(colors)]
        geometry = {
            'archetype': i + 1,
            'width': np.nan,
            'height': np.nan,
            'angle': np.nan,
        }

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            print(f"  Skipping confidence ellipse for archetype {i + 1}: invalid mean/covariance")
            ellipse_geometry.append(geometry)
            continue
        # compute eigenvalues and eigenvectors of the covariance matrix
        #eigenvalues = amount of spread along ellipse axes
        #eigenvectors = directions of ellipse axes
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = eigvals.argsort()[::-1]

        eigvals = eigvals[order]
        eigvecs = eigvecs[:,order]

        if np.any(eigvals < -1e-10):
            print(f"  Skipping confidence ellipse for archetype {i + 1}: negative covariance eigenvalues {eigvals}")
            ellipse_geometry.append(geometry)
            continue
        eigvals = np.clip(eigvals, 0, None)

        angle = np.degrees(
            np.arctan2(eigvecs[1,0], eigvecs[0,0])
        )

        width = 2 * scale * np.sqrt(eigvals[0])
        height = 2 * scale * np.sqrt(eigvals[1])
        geometry.update({
            'width': width,
            'height': height,
            'angle': angle,
        })
        ellipse_geometry.append(geometry)

        ellipse = Ellipse(mean,
                          width,
                          height,
                          angle=angle,
                          edgecolor=color,
                          facecolor=to_rgba(color, 0.18),
                          lw=2)

        ax.add_patch(ellipse)

        ax.text(mean[0],
                mean[1],
                str(i+1),
                fontsize=14,
                weight='bold',
                color=color)

    ax.set_aspect("equal")
    ax.set_title("Mean archetype triangle")

    plt.tight_layout()
    _disable_rasterization(plt.gcf())
    plt.savefig(output_file, dpi=300)
    plt.close()
    ellipse_geometry_df = pd.DataFrame(ellipse_geometry)
    return mean_coords_df, ellipse_geometry_df


def plot_mean_hormone_archetype_pc1_pc2_by_sex(mean_hormones_arch, directory_path,
                                               mean_coords_df=None, sex_col='sex'):
    required_cols = {'PC1', 'PC2', sex_col}
    missing_cols = sorted(required_cols - set(mean_hormones_arch.columns))
    if missing_cols:
        print(f'  Skipped mean hormone archetype PC plot; missing columns: {missing_cols}')
        return

    coords_path = os.path.join(directory_path, 'mean_archetype_coordinates.xlsx')
    ellipse_geometry_df = pd.DataFrame()
    if os.path.exists(coords_path):
        try:
            mean_coords_df = pd.read_excel(coords_path, sheet_name='mean_coordinates')
            ellipse_geometry_df = pd.read_excel(coords_path, sheet_name='ellipse_geometry')
        except ValueError:
            mean_coords_df = pd.read_excel(coords_path)
            print(f'  Loaded mean archetype coordinates without ellipse sheet: {coords_path}')
        else:
            print(f'  Loaded mean archetype coordinates and ellipses: {coords_path}')
    elif mean_coords_df is None:
        print(f'  Skipped mean hormone archetype PC plot; missing file: {coords_path}')
        return

    if mean_coords_df is None or not {'PC1', 'PC2'}.issubset(mean_coords_df.columns):
        print('  Skipped mean hormone archetype PC plot; missing mean PC coordinates')
        return

    plot_df = mean_hormones_arch.dropna(subset=['PC1', 'PC2']).copy()
    if plot_df.empty:
        print('  Skipped mean hormone archetype PC plot; no finite hormone archetype PC coordinates')
        return

    mean_coords = mean_coords_df.copy()
    if 'archetype' not in mean_coords.columns:
        mean_coords.insert(0, 'archetype', np.arange(1, len(mean_coords) + 1))
    mean_coords = mean_coords.dropna(subset=['PC1', 'PC2']).copy()
    mean_coords = mean_coords.sort_values('archetype')

    fig, ax = plt.subplots(figsize=(8, 8))
    sex_values = plot_df[sex_col].astype(str).str.strip().str.lower()
    female_mask = sex_values.isin(['f', 'female'])
    male_mask = sex_values.isin(['m', 'male'])
    unknown_mask = ~(female_mask | male_mask)

    if female_mask.any():
        ax.plot(plot_df.loc[female_mask, 'PC1'], plot_df.loc[female_mask, 'PC2'],
                linestyle='None', marker='o', markersize=10, markerfacecolor='red',
                markeredgecolor='white', markeredgewidth=0.6, label='Female')
    if male_mask.any():
        ax.plot(plot_df.loc[male_mask, 'PC1'], plot_df.loc[male_mask, 'PC2'],
                linestyle='None', marker='o', markersize=10, markerfacecolor='royalblue',
                markeredgecolor='white', markeredgewidth=0.6, label='Male')
    if unknown_mask.any():
        ax.plot(plot_df.loc[unknown_mask, 'PC1'], plot_df.loc[unknown_mask, 'PC2'],
                linestyle='None', marker='o', markersize=5, markerfacecolor='gray',
                markeredgecolor='white', markeredgewidth=0.6, label='Other/unknown')

    arch_colors = ['tomato', 'seagreen', 'royalblue']
    if len(mean_coords) >= 3:
        triangle = mean_coords.iloc[:3]
        closed = np.vstack([triangle[['PC1', 'PC2']].to_numpy(dtype=float),
                            triangle[['PC1', 'PC2']].iloc[0].to_numpy(dtype=float)])
        ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=2.5,
                label='Mean archetype triangle')

    for _, row in mean_coords.iterrows():
        archetype = int(row['archetype']) if pd.notna(row['archetype']) else None
        color = arch_colors[(archetype - 1) % len(arch_colors)] if archetype is not None else 'black'
        # ax.scatter(row['PC1'], row['PC2'], marker='^', s=160,
        #            facecolor='white', edgecolor=color, linewidth=2.5, zorder=5)
        ax.text(row['PC1'], row['PC2'], str(archetype), fontsize=13, weight='bold',
                color=color, ha='center', va='center', zorder=6)

    if not ellipse_geometry_df.empty:
        mean_lookup = mean_coords.set_index('archetype')
        for _, ellipse_row in ellipse_geometry_df.iterrows():
            archetype = ellipse_row.get('archetype')
            if pd.isna(archetype) or archetype not in mean_lookup.index:
                continue

            try:
                width, height, angle = np.asarray(
                    [ellipse_row.get('width'), ellipse_row.get('height'), ellipse_row.get('angle')],
                    dtype=float,
                )
            except (TypeError, ValueError):
                continue

            if not np.all(np.isfinite([width, height, angle])):
                continue

            mean = mean_lookup.loc[archetype]
            color = arch_colors[(int(archetype) - 1) % len(arch_colors)]
            ellipse = Ellipse((mean['PC1'], mean['PC2']), width, height, angle=angle,
                              edgecolor=color, facecolor=to_rgba(color, 0.16), lw=2)
            ax.add_patch(ellipse)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Mean hormone archetype PC1/PC2 by sex')
    ax.set_aspect('equal')
    ax.margins(0.12)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', which='both', direction='out', length=4, width=1,
                   color='black')
    ax.legend(fontsize=8)

    output_path = os.path.join(directory_path, 'mean_hormones_archetype_pc1_pc2_by_sex.pdf')
    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved mean hormone archetype PC plot by sex: {output_path}')


def plot_mean_hormone_feature_correlations_with_archetype_probs(mean_hormones_arch,
                                                                metadata_cols,
                                                                directory_path):
    target_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    missing_cols = [col for col in target_cols if col not in mean_hormones_arch.columns]
    if missing_cols:
        print(f'  Skipped mean hormone feature correlations; missing columns: {missing_cols}')
        return None

    excluded_cols = set((metadata_cols or []) + [
        'PC1', 'PC2', 'Dominant_archetype',
        'Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob',
    ])
    feature_cols = [
        col for col in mean_hormones_arch.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(mean_hormones_arch[col])
    ]
    if not feature_cols:
        print('  Skipped mean hormone feature correlations; no numeric feature columns found')
        return None

    rows = []
    for target_col in target_cols:
        archetype = target_col.replace('_prob', '')
        for feature_col in feature_cols:
            pair_df = mean_hormones_arch[[feature_col, target_col]].dropna()
            if len(pair_df) < 3 or pair_df[feature_col].nunique() < 2 or pair_df[target_col].nunique() < 2:
                r_value = np.nan
                p_value = np.nan
            else:
                r_value, p_value = pearsonr(pair_df[feature_col], pair_df[target_col])

            rows.append({
                'Feature': feature_col,
                'Archetype': archetype,
                'Pearson_r': r_value,
                'p_value': p_value,
                'n': len(pair_df),
            })

    corr_df = pd.DataFrame(rows)
    corr_df['p_adjusted_BH'] = np.nan
    for archetype in corr_df['Archetype'].unique():
        mask = corr_df['Archetype'] == archetype
        valid_pvalues = mask & corr_df['p_value'].notna()
        if valid_pvalues.any():
            _, adjusted_pvalues, _, _ = multipletests(
                corr_df.loc[valid_pvalues, 'p_value'],
                method='fdr_bh',
            )
            corr_df.loc[valid_pvalues, 'p_adjusted_BH'] = adjusted_pvalues

    output_excel = os.path.join(directory_path, 'mean_hormone_feature_archetype_probability_correlations.xlsx')
    corr_df.to_excel(output_excel, index=False)

    archetypes = [col.replace('_prob', '') for col in target_cols]
    archetype_titles = {
        'Archetype1': 'Archetype 1',
        'Archetype2': 'Archetype 2',
        'Archetype3': 'Archetype 3',
    }
    plot_df = corr_df[
        corr_df['p_adjusted_BH'].notna()
        & (corr_df['p_adjusted_BH'] < 0.1)
        & corr_df['Pearson_r'].notna()
    ].copy()
    max_rows = max(1, *(len(plot_df[plot_df['Archetype'] == archetype]) for archetype in archetypes))
    max_abs_corr = plot_df['Pearson_r'].abs().max() if not plot_df.empty else 0.5
    x_limit = min(1.0, max(0.5, np.ceil((max_abs_corr + 0.05) * 10) / 10))

    fig, axes = plt.subplots(
        1, 3,
        figsize=(9.2, max(3.0, 0.38 * max_rows + 1.2)),
        sharex=True,
    )
    point_color = '#7f8fa0'
    line_color = '#5f6871'
    faint_alpha = 0.32

    for ax, archetype in zip(axes, archetypes):
        arch_df = plot_df[plot_df['Archetype'] == archetype].copy()
        arch_df = arch_df.sort_values('Pearson_r', ascending=True).reset_index(drop=True)

        if arch_df.empty:
            ax.text(0.5, 0.5, 'No BH < 0.1\ncorrelations', ha='center', va='center',
                    fontsize=8, transform=ax.transAxes)
            ax.set_yticks([])
        else:
            y_positions = np.arange(len(arch_df))
            for y_pos, row in zip(y_positions, arch_df.itertuples(index=False)):
                alpha = faint_alpha if row.p_adjusted_BH > 0.05 else 1.0
                ax.hlines(y_pos, 0, row.Pearson_r, color=line_color, linewidth=1.2, alpha=alpha)
                ax.scatter(row.Pearson_r, y_pos, s=70, color=point_color, alpha=alpha,
                           edgecolors='none', zorder=3)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(arch_df['Feature'], fontsize=7)

        ax.set_ylim(max_rows - 0.4, -0.6)
        ax.axvline(0, color='black', linewidth=1.0)
        ax.axhline(-0.5, color='#808080', linewidth=0.8)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', labelsize=7, length=0, pad=1, labeltop=True, labelbottom=False)
        ax.tick_params(axis='y', length=0)
        ax.set_title(archetype_titles.get(archetype, archetype), fontsize=9, fontweight='bold', pad=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.5, 0.02, 'Pearson r with archetype probability; transparent markers indicate 0.05 < BH < 0.1',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.2)

    output_pdf = os.path.join(directory_path, 'mean_hormone_feature_archetype_probability_correlations.pdf')
    _disable_rasterization(fig)
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'  Saved mean hormone feature archetype probability correlations: {output_excel}')
    print(f'  Saved mean hormone feature archetype probability correlation plot: {output_pdf}')
    return corr_df


def plot_mean_behavior_feature_correlations_with_mean_pcs(behavior_df,
                                                          mean_table_df,
                                                          metadata_cols,
                                                          directory_path):
    target_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    missing_cols = [col for col in target_cols if col not in mean_table_df.columns]
    if missing_cols:
        print(f'  Skipped mean behavior feature archetype probability correlations; missing columns: {missing_cols}')
        return None

    correlation_df = pd.concat(
        [behavior_df.reset_index(drop=True), mean_table_df[target_cols].reset_index(drop=True)],
        axis=1,
    )

    excluded_cols = set((metadata_cols or []) + [
        'PC1', 'PC2', 'Dominant_archetype',
        'Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob',
    ])
    feature_cols = [
        col for col in correlation_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(correlation_df[col])
    ]
    if not feature_cols:
        print('  Skipped mean behavior feature archetype probability correlations; no numeric feature columns found')
        return None

    rows = []
    for arch_idx, target_col in enumerate(target_cols, start=1):
        for feature_col in feature_cols:
            pair_df = correlation_df[[feature_col, target_col]].dropna()
            if len(pair_df) < 3 or pair_df[feature_col].nunique() < 2 or pair_df[target_col].nunique() < 2:
                r_value = np.nan
                p_value = np.nan
            else:
                r_value, p_value = pearsonr(pair_df[feature_col], pair_df[target_col])

            rows.append({
                'Feature': feature_col,
                'Archetype': f'Archetype{arch_idx}',
                'Pearson_r': r_value,
                'p_value': p_value,
                'n': len(pair_df),
            })

    corr_df = pd.DataFrame(rows)
    corr_df['p_adjusted_BH'] = np.nan
    for archetype in corr_df['Archetype'].unique():
        mask = corr_df['Archetype'] == archetype
        valid_pvalues = mask & corr_df['p_value'].notna()
        if valid_pvalues.any():
            _, adjusted_pvalues, _, _ = multipletests(
                corr_df.loc[valid_pvalues, 'p_value'],
                method='fdr_bh',
            )
            corr_df.loc[valid_pvalues, 'p_adjusted_BH'] = adjusted_pvalues

    corr_df = corr_df.sort_values(['Archetype', 'p_adjusted_BH'], na_position='last')
    significant_df = corr_df[corr_df['p_adjusted_BH'] < 0.05].copy()

    output_excel = os.path.join(directory_path, 'mean_behavior_feature_archetype_probability_correlations.xlsx')
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='all_correlations', index=False)
        significant_df.to_excel(writer, sheet_name='BH_p_lt_0_05', index=False)

    archetypes = ['Archetype1', 'Archetype2', 'Archetype3']
    max_rows = 1
    if not significant_df.empty:
        max_rows = max(max_rows, *(len(significant_df[significant_df['Archetype'] == arch])
                                  for arch in archetypes))

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.28 * max_rows)), sharex=True)
    colors = ['tomato', 'seagreen', 'royalblue']

    for ax, archetype, color in zip(axes, archetypes, colors):
        arch_df = significant_df[significant_df['Archetype'] == archetype].copy()
        arch_df = arch_df.sort_values('Pearson_r', ascending=True)

        if arch_df.empty:
            ax.text(0.5, 0.5, 'No BH-significant\ncorrelations', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_yticks([])
        else:
            ax.barh(arch_df['Feature'], arch_df['Pearson_r'], color=color, alpha=0.75)
            ax.tick_params(axis='y', labelsize=10)

        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(archetype)
        ax.set_xlabel('Pearson r')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('BH-significant behaviour correlations with mean archetype probabilities (p < 0.05)')
    fig.tight_layout()

    output_pdf = os.path.join(directory_path, 'mean_behavior_feature_archetype_probability_correlations_BH_p_lt_0_05.pdf')
    _disable_rasterization(fig)
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'  Saved mean behavior feature archetype probability correlations: {output_excel}')
    print(f'  Saved mean behavior feature archetype probability correlation plot: {output_pdf}')
    return significant_df


def plot_average_confusion_matrices(accepted_results, output_file, model_names=None, labels=None):
    if not accepted_results:
        return

    if model_names is None:
        model_names = list(accepted_results[0]['loocv_results'].keys())
    if labels is None:
        n_classes = accepted_results[0]['archetypes'].shape[0]
        labels = list(range(1, n_classes + 1))

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        normalized_cms = []

        for result in accepted_results:
            model_result = result['loocv_results'].get(model_name)
            if model_result is None:
                continue

            cm = confusion_matrix(result['aligned_true'], model_result['predictions'], labels=labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) #if there are no samples for a class, we avoid division by zero and set the normalized values to zero
            normalized_cms.append(cm_normalized)

        if not normalized_cms:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(model_name)
            continue

        normalized_cms = np.asarray(normalized_cms)#convert list of normalized confusion matrices to a numpy array for easier computation first is the number of iterations, second is the number of classes, third is the number of classes
        mean_cm = np.mean(normalized_cms, axis=0) #on the number of iterations axis, we compute the mean confusion matrix across all iterations
        std_cm = np.std(normalized_cms, axis=0)

        text_labels = [[f'{mean_cm[row, col] * 100:.1f}%\n±{std_cm[row, col] * 100:.1f}%'
                        for col in range(mean_cm.shape[1])]
                       for row in range(mean_cm.shape[0])]
        _draw_vector_matrix(ax, mean_cm, labels=labels, cmap='Blues',
                            vmin=0, vmax=1, text_labels=text_labels, text_size=8)

        ax.set_title(f'{model_name} average confusion\n(n={len(normalized_cms)})', fontsize=10)

    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved average confusion matrices: {output_file}')
