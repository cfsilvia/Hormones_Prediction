import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import os


def setup_figure(n_models=4):
    fig = plt.figure(figsize=(26, 6))
    gs = GridSpec(1, n_models + 1, figure=fig)
    ax_tri = fig.add_subplot(gs[0, 0])
    model_names = ['LogReg', 'SVM', 'MLP', 'XGBoost'][:n_models]
    cm_axes = {}
    for i, name in enumerate(model_names):
        cm_axes[name] = fig.add_subplot(gs[0, i + 1])
    return fig, ax_tri, cm_axes


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
            normalize='true', values_format='.0%', cmap='Blues',
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
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cm_normalized, display_labels=[1, 2, 3]).plot(
            ax=ax, cmap='Blues', values_format='.0%', colorbar=False, text_kw={'fontsize': 9})
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
