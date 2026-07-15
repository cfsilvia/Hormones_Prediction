import json
import os
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'pdf.use14corefonts': False,
    'image.composite_image': False,
})
import pandas as pd
import numpy as np
from archetype_analysis import (compute_pca, predict_archetype_loocv, predict_archetype_loocv_with_shap,
    sample_and_fit_archetypes, compute_archetype_probabilities, build_mean_archetype_assignment)
from alignment.label_alignment import accumulate_results, apply_mapping
from plot_utils import (plot_loocv_confusion_matrices, calculate_loocv_permutation_pvalues, plot_mean_archetype_triangle, plot_average_confusion_matrices,
                        setup_figure, draw_triangle, plot_confusion_matrices,
                        plot_mean_hormone_archetype_pc1_pc2_by_sex,
                        plot_mean_hormone_feature_correlations_with_archetype_probs,
                        plot_mean_behavior_feature_correlations_with_mean_pcs)
from sklearn.metrics import confusion_matrix, f1_score
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from main_utils import (
    load_data, save_behavior_parameters, setup_iteration_state,
    build_table_df, select_top_and_merge_hormones,
    predict_and_align, compute_iteration_metrics, save_classification_metrics,
)


_WORKER_CONTEXT = {}


def _disable_rasterization(fig):
    for artist in fig.findobj():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(False)


def get_alignment_mapping(reference_archetypes, current_archetypes):
    n_arch = reference_archetypes.shape[0]
    dists = np.zeros((n_arch, n_arch))
    for i in range(n_arch):
        for j in range(n_arch):
            dists[i, j] = np.linalg.norm(reference_archetypes[i] - current_archetypes[j])

    row_ind, col_ind = linear_sum_assignment(dists)
    mapping = np.zeros(n_arch, dtype=int)
    for ref_idx, curr_idx in zip(row_ind, col_ind):
        mapping[curr_idx] = ref_idx
    return mapping


def _init_iteration_worker(all_pca_coords, sample_frac, no_pca_data, behavior_df,
                           metadata_cols, hormones_df, if_dominant_archetype,
                           user_models, min_per_vertex, n_permutations):
    np.random.seed()
    _WORKER_CONTEXT.update(
        all_pca_coords=all_pca_coords,
        sample_frac=sample_frac,
        no_pca_data=no_pca_data,
        behavior_df=behavior_df,
        metadata_cols=metadata_cols,
        hormones_df=hormones_df,
        if_dominant_archetype=if_dominant_archetype,
        user_models=user_models,
        min_per_vertex=min_per_vertex,
        n_permutations=n_permutations,
    )


def _compute_permutation_results(hormones_arch, metadata_cols, loocv_results,
                                 aligned_true, model_names, n_permutations):
    from archetype_analysis import _get_models
    from plot_utils import permutation_test_significance

    feature_cols = [c for c in hormones_arch.columns
                    if c not in metadata_cols and c != 'Dominant_archetype'
                    and c != 'PC1' and c != 'PC2']
    X = hormones_arch[feature_cols].select_dtypes(include=[np.number]).values
    model_lookup = _get_models(n_classes=3, model_names=model_names)

    permutation_results = {}
    for mname, mres in loocv_results.items():
        labels = np.unique(aligned_true)
        observed = f1_score(aligned_true, mres['predictions'], average='macro')
        observed_per_class = f1_score(aligned_true, mres['predictions'], labels=labels,
                                      average=None, zero_division=0)
        null_scores, _, p_val = permutation_test_significance(
            X, aligned_true, model_lookup[mname], mname, observed, n_permutations)
        permutation_results[mname] = dict(
            p_value=p_val,
            observed=observed,
            observed_per_class=observed_per_class,
            null_scores=null_scores,
            null_mean=np.mean(null_scores),
        )
    return permutation_results


def _run_iteration_candidate(attempt_number):
    ctx = _WORKER_CONTEXT
    all_pca_coords = ctx['all_pca_coords']

    _, archetypes, varexlp, counts, sample_indices = sample_and_fit_archetypes(
        all_pca_coords, ctx['sample_frac'], ctx['no_pca_data'])
    idx_tuple = tuple(sorted(int(i) for i in sample_indices))

    if not all(c >= ctx['min_per_vertex'] for c in counts):
        return dict(
            accepted=False,
            attempt=attempt_number,
            idx_tuple=idx_tuple,
            counts=counts,
            varexlp=varexlp,
        )

    prob_coeffs = compute_archetype_probabilities(all_pca_coords, archetypes)
    table_df = build_table_df(
        ctx['behavior_df'], ctx['metadata_cols'], all_pca_coords, prob_coeffs)
    hormones_arch = select_top_and_merge_hormones(
        table_df, archetypes, ctx['hormones_df'], ctx['if_dominant_archetype'])
    loocv_results, aligned_true = predict_and_align(
        hormones_arch, ctx['metadata_cols'], ctx['if_dominant_archetype'], None,
        model_names=ctx['user_models'])

    metrics = compute_iteration_metrics(aligned_true, loocv_results)
    permutation_results = _compute_permutation_results(
        hormones_arch, ctx['metadata_cols'], loocv_results, aligned_true,
        ctx['user_models'], ctx['n_permutations'])
    
    pvalues = {mname: result['p_value'] for mname, result in permutation_results.items()}
    for metric in metrics:
        metric['p_value'] = pvalues.get(metric['model'])

    return dict(
        accepted=True,
        attempt=attempt_number,
        idx_tuple=idx_tuple,
        archetypes=archetypes,
        varexlp=varexlp,
        counts=counts,
        loocv_results=loocv_results,
        aligned_true=aligned_true,
        metrics=metrics,
        pvalues=pvalues,
        permutation_results=permutation_results,
    )


def _plot_saved_permutation_tests(perm_axes, permutation_results):
    for mname, result in permutation_results.items():
        ax = perm_axes[mname]
        ax.clear()
        null_scores = result['null_scores']
        observed = result['observed']
        p_value = result['p_value']
        observed_per_class = result['observed_per_class']

        ax.hist(null_scores, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)
        ax.axvline(observed, color='red', linewidth=2, label=f'Observed: {observed:.3f}')
        ax.axvline(np.percentile(null_scores, 95), color='orange', linestyle='--', label='95th percentile')
        ax.set_xlabel('F1 macro')
        ax.set_ylabel('Density')
        per_class_text = ' '.join(f'F1_{i + 1}={score:.3f}'
                                  for i, score in enumerate(observed_per_class))
        ax.set_title(f'{mname}\np={p_value:.4f} F1={observed:.3f}\n{per_class_text}', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def _apply_reference_mapping(result, mapping):
    result['mapping'] = mapping
    for model_result in result['loocv_results'].values():
        model_result['predictions'] = apply_mapping(model_result['predictions'], mapping)
    result['aligned_true'] = apply_mapping(result['aligned_true'], mapping)
    result['metrics'] = compute_iteration_metrics(result['aligned_true'], result['loocv_results'])
    for metric in result['metrics']:
        metric['p_value'] = result['pvalues'].get(metric['model'])

    labels = np.arange(1, len(mapping) + 1)
    for model_name, model_result in result['loocv_results'].items():
        if model_name in result['permutation_results']:
            result['permutation_results'][model_name]['observed_per_class'] = f1_score(
                result['aligned_true'], model_result['predictions'], labels=labels,
                average=None, zero_division=0)


def _save_iteration_pdf(directory_path, all_pca_coords, no_pca_data, exclude_pca_data,
                        user_models, result):
    output_dir = os.path.join(directory_path, 'iteration_plots')
    os.makedirs(output_dir, exist_ok=True)

    fig, ax_tri, cm_axes, perm_axes = setup_figure(model_names=user_models)
    if exclude_pca_data:
        coords_to_remove = no_pca_data[['PC1', 'PC2']].to_numpy(dtype=float)
        matches = np.isclose(all_pca_coords[:, None, :], coords_to_remove[None, :, :]).all(axis=2).any(axis=1)
        all_pca_coords_plot = all_pca_coords[~matches]
    else:
        all_pca_coords_plot = all_pca_coords

    draw_triangle(ax_tri, all_pca_coords_plot, result['archetypes'], result['counts'],
                  result['varexlp'], result['iteration'], result['attempt'],
                  mapping=result.get('mapping'))
    plot_confusion_matrices(cm_axes, result['loocv_results'], result['aligned_true'])
    _plot_saved_permutation_tests(perm_axes, result['permutation_results'])

    fig.tight_layout()
    pvalue_text = '_'.join(
        f'{mname}_p{result["pvalues"][mname]:.4f}'
        for mname in user_models if mname in result['pvalues'])
    output_path = os.path.join(output_dir, f'iteration_{result["iteration"]:03d}_{pvalue_text}.pdf')
    _disable_rasterization(fig)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved iteration plot: {output_path}')


def _save_parallel_iteration_outputs(directory_path, accepted_results):
    prediction_rows = []
    archetype_rows = []
    mapping_rows = []
    pvalue_rows = []

    for result in accepted_results:
        iteration = result['iteration']
        mapping = result.get('mapping')
        for vertex, coords in enumerate(result['archetypes'], start=1):
            reference_archetype = int(mapping[vertex - 1]) + 1 if mapping is not None else None
            archetype_rows.append(dict(
                iteration=iteration,
                vertex=vertex,
                reference_archetype=reference_archetype,
                PC1=coords[0],
                PC2=coords[1],
                count=result['counts'][vertex - 1],
                variance_explained=result['varexlp'],
                attempt=result['attempt'],
            ))
            mapping_rows.append(dict(
                iteration=iteration,
                current_archetype=vertex,
                reference_archetype=reference_archetype,
            ))

        for model_name, pvalue in result['pvalues'].items():
            pvalue_rows.append(dict(
                iteration=iteration,
                model=model_name,
                p_value=pvalue,
            ))

        for model_name, model_result in result['loocv_results'].items():
            for row_idx, (true_label, pred_label) in enumerate(
                    zip(result['aligned_true'], model_result['predictions'])):
                prediction_rows.append(dict(
                    iteration=iteration,
                    row=row_idx,
                    model=model_name,
                    true_label=true_label,
                    predicted_label=pred_label,
                    accuracy=model_result['accuracy'],
                ))

    output_path = os.path.join(directory_path, 'parallel_iteration_results.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(archetype_rows).to_excel(writer, sheet_name='archetypes', index=False)
        pd.DataFrame(mapping_rows).to_excel(writer, sheet_name='archetype_mapping', index=False)
        pd.DataFrame(pvalue_rows).to_excel(writer, sheet_name='pvalues', index=False)
        pd.DataFrame(prediction_rows).to_excel(writer, sheet_name='loocv_predictions', index=False)
    print(f'  Saved parallel iteration results: {output_path}')


def _plot_permutation_mean_vs_real_fscore(directory_path, accepted_results, model_names):
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(6, 3.5 * n_models), squeeze=False)

    for row_idx, model_name in enumerate(model_names):
        random_mean_fscores = []
        real_fscores = []

        for result in accepted_results:
            permutation_result = result.get('permutation_results', {}).get(model_name)
            if permutation_result is not None:
                null_mean = permutation_result.get('null_mean')
                if null_mean is None:
                    null_mean = np.mean(permutation_result['null_scores'])
                random_mean_fscores.append(float(null_mean))

            metric_f1 = next(
                (metric['f1_macro'] for metric in result.get('metrics', [])
                 if metric['model'] == model_name),
                None)
            if metric_f1 is not None:
                real_fscores.append(float(metric_f1))

        ax = axes[row_idx, 0]
        all_values = random_mean_fscores + real_fscores
        if all_values:
            bins = min(20, max(5, len(all_values)))
            density = len(all_values) > 1
            if random_mean_fscores:
                ax.hist(random_mean_fscores, bins=bins, alpha=0.6, color='gray',
                        edgecolor='black', density=density,
                        label='Mean random permutation F1')
                ax.axvline(np.mean(random_mean_fscores), color='black', linestyle='--',
                           linewidth=2, label=f'Random mean: {np.mean(random_mean_fscores):.3f}')
            if real_fscores:
                ax.hist(real_fscores, bins=bins, alpha=0.6, color='steelblue',
                        edgecolor='black', density=density,
                        label='Real F1')
                ax.axvline(np.mean(real_fscores), color='red', linewidth=2,
                           label=f'Real mean: {np.mean(real_fscores):.3f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title(f'{model_name} - random permutation vs real F1', fontsize=10)
        ax.set_xlabel('F1 macro')
        ax.set_ylabel('Density' if len(all_values) > 1 else 'Count')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(directory_path, 'permutation_mean_vs_real_fscore.pdf')
    _disable_rasterization(fig)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved permutation mean vs real F-score plot: {output_path}')


def _safe_excel_sheet_name(name, used_names):
    invalid_chars = '[]:*?/\\'
    safe_name = ''.join('_' if char in invalid_chars else char for char in str(name))[:31]
    safe_name = safe_name or 'Sheet'
    base_name = safe_name
    suffix = 1
    while safe_name in used_names:
        suffix_text = f'_{suffix}'
        safe_name = f'{base_name[:31 - len(suffix_text)]}{suffix_text}'
        suffix += 1
    used_names.add(safe_name)
    return safe_name


def _shap_feature_importance_rows(model_name, feature_cols, shap_values):
    shap_array = np.asarray(shap_values, dtype=float)
    mean_abs_shap = np.nanmean(np.abs(shap_array), axis=0)
    mean_shap = np.nanmean(shap_array, axis=0)
    total_importance = np.nansum(mean_abs_shap)
    importance_percent = (
        mean_abs_shap / total_importance * 100 if total_importance else np.zeros_like(mean_abs_shap)
    )

    rows = []
    for feature, abs_value, percent, signed_value in zip(
            feature_cols, mean_abs_shap, importance_percent, mean_shap):
        if signed_value > 0:
            direction = 'increases prediction'
        elif signed_value < 0:
            direction = 'decreases prediction'
        else:
            direction = 'no average direction'

        rows.append({
            'model_name': model_name,
            'feature': feature,
            'mean_abs_SHAP': abs_value,
            'importance_percent': percent,
            'mean_SHAP': signed_value,
            'direction': direction,
        })
    return rows


def _mean_archetype_metric_rows(loocv_results):
    rows = []
    for model_name, model_result in loocv_results.items():
        true_labels = np.asarray(model_result['true_labels'])
        predicted_labels = np.asarray(model_result['predictions'])

        class_keys = list(model_result.get('f1_per_class', {}).keys())
        if not class_keys:
            class_keys = [str(label.item() if hasattr(label, 'item') else label)
                          for label in pd.unique(true_labels)]
        #convert true_labels and predicted_labels to strings for confusion_matrix
        true_keys = [str(label.item() if hasattr(label, 'item') else label) for label in true_labels]
        predicted_keys = [str(label.item() if hasattr(label, 'item') else label) for label in predicted_labels]
        cm = confusion_matrix(true_keys, predicted_keys, labels=class_keys)

        row = {
            'model_name': model_name,
            'confusion_matrix': json.dumps(cm.tolist()),
            'accuracy': model_result.get('accuracy'),
            'fmacro': model_result.get('f1_macro'),
            'roc_auc': model_result.get('roc_auc_ovr_macro', model_result.get('roc_auc')),
        }

        for class_key in class_keys:
            row[f'fscore_class_{class_key}'] = model_result.get('f1_per_class', {}).get(class_key)
            row[f'precision_class_{class_key}'] = model_result.get('precision_per_class', {}).get(class_key)
            row[f'recall_class_{class_key}'] = model_result.get('recall_per_class', {}).get(class_key)
            row[f'roc_auc_class_{class_key}'] = model_result.get('roc_auc_per_class', {}).get(class_key)

        rows.append(row)
    return rows


def save_mean_archetype_shap_values(loocv_results, source_df, directory_path):
    metrics_path = os.path.join(directory_path, 'metrics_for_mean_archetypes.xlsx')
    metrics_rows = _mean_archetype_metric_rows(loocv_results)
    with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name='metrics', index=False)
    print(f'  Saved mean archetype metrics: {metrics_path}')

    output_path = os.path.join(directory_path, 'shap_values_for_mean_archetypes.xlsx')
    used_sheet_names = set() #a set to keep track of used sheet names to avoid duplicates

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        importance_rows = []
        for model_name, model_result in loocv_results.items():
            feature_cols = model_result['feature_cols']
            importance_rows.extend(_shap_feature_importance_rows(
                model_name, feature_cols, model_result['shap_values'],
            ))
            metadata_df = model_result['metadata'].reset_index(drop=True)
            feature_df = source_df[feature_cols].reset_index(drop=True).add_prefix('feature_')
            shap_df = pd.DataFrame(
                model_result['shap_values'],
                columns=[f'shap_{feature}' for feature in feature_cols],
            )
            labels_df = pd.DataFrame({
                'true_label': model_result['true_labels'],
                'predicted_label': model_result['predictions'],
            })
            model_df = pd.concat([metadata_df, labels_df, feature_df, shap_df], axis=1)
            sheet_name = _safe_excel_sheet_name(model_name, used_sheet_names)
            model_df.to_excel(writer, sheet_name=sheet_name, index=False)

            shap_all_classes = _shap_all_classes_array(model_result)
            if shap_all_classes is not None:
                n_features = len(feature_cols)
                if shap_all_classes.shape[1] != n_features and shap_all_classes.shape[2] == n_features:
                    shap_all_classes = np.transpose(shap_all_classes, (0, 2, 1))

                if shap_all_classes.shape[1] == n_features:
                    class_labels = np.unique(np.asarray(model_result.get('true_labels', [])))
                    n_classes = shap_all_classes.shape[2]
                    if len(class_labels) != n_classes:
                        class_labels = np.arange(1, n_classes + 1)

                    for class_idx, class_label in enumerate(class_labels):
                        class_shap_df = pd.DataFrame(
                            shap_all_classes[:, :, class_idx],
                            columns=[f'shap_{feature}' for feature in feature_cols],
                        )
                        class_df = pd.concat([metadata_df, labels_df, feature_df, class_shap_df], axis=1)
                        class_sheet = _safe_excel_sheet_name(
                            f'{model_name}_{_archetype_title(class_label)}', used_sheet_names,
                        )
                        class_df.to_excel(writer, sheet_name=class_sheet, index=False)
                else:
                    print(f'  Skipped class-specific SHAP sheets for {model_name}; SHAP feature shape mismatch')

        if importance_rows:
            importance_df = pd.DataFrame(importance_rows).sort_values(
                ['model_name', 'mean_abs_SHAP'], ascending=[True, False],
            )
            importance_sheet = _safe_excel_sheet_name('feature_importance', used_sheet_names)
            importance_df.to_excel(writer, sheet_name=importance_sheet, index=False)

    print(f'  Saved mean archetype SHAP values: {output_path}')


def plot_mean_archetype_shap_importance(loocv_results, directory_path, sex_col='sex'):
    global_path = os.path.join(directory_path, 'mean_archetype_shap_global_importance.pdf')
    sex_path = os.path.join(directory_path, 'mean_archetype_shap_importance_by_sex.pdf')

    n_models = len(loocv_results)
    if n_models == 0:
        return

    max_features = max(len(model_result['feature_cols']) for model_result in loocv_results.values())
    fig_height = max(4, 0.35 * max_features * n_models)

    fig, axes = plt.subplots(n_models, 1, figsize=(10, fig_height), squeeze=False)
    for ax, (model_name, model_result) in zip(axes.ravel(), loocv_results.items()):
        feature_cols = np.asarray(model_result['feature_cols'])
        shap_array = np.asarray(model_result['shap_values'], dtype=float)
        mean_abs_shap = np.nanmean(np.abs(shap_array), axis=0)
        total_importance = np.nansum(mean_abs_shap)
        importance_percent = (
            mean_abs_shap / total_importance * 100
            if total_importance else np.zeros_like(mean_abs_shap)
        )
        order = np.argsort(importance_percent)

        ax.barh(np.arange(len(order)), importance_percent[order], color='steelblue')
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(feature_cols[order], fontsize=8)
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'{model_name} global SHAP importance (%)')
        ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(global_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved mean archetype global SHAP importance plot: {global_path}')

    if any(sex_col not in model_result['metadata'].columns for model_result in loocv_results.values()):
        print(f'  Skipped SHAP importance by sex plot; missing column: {sex_col}')
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(10, fig_height), squeeze=False)
    for ax, (model_name, model_result) in zip(axes.ravel(), loocv_results.items()):
        feature_cols = np.asarray(model_result['feature_cols'])
        shap_array = np.asarray(model_result['shap_values'], dtype=float)
        abs_shap = np.nan_to_num(np.abs(shap_array), nan=0.0)
        sex_values = model_result['metadata'][sex_col].reset_index(drop=True).astype(str).str.lower()
        female_mask = sex_values.isin(['f', 'female']).to_numpy()
        male_mask = sex_values.isin(['m', 'male']).to_numpy()
        known_count = int(female_mask.sum() + male_mask.sum())

        if known_count == 0:
            ax.text(0.5, 0.5, 'No male/female labels', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_axis_off()
            continue

        female_contribution = abs_shap[female_mask].sum(axis=0) / known_count
        male_contribution = abs_shap[male_mask].sum(axis=0) / known_count
        total_importance = np.nansum(female_contribution + male_contribution)
        if total_importance:
            female_contribution = female_contribution / total_importance * 100
            male_contribution = male_contribution / total_importance * 100
        order = np.argsort(female_contribution + male_contribution)
        y_pos = np.arange(len(order))

        ax.barh(y_pos, female_contribution[order], color='tomato', label='Female')
        ax.barh(y_pos, male_contribution[order], left=female_contribution[order],
                color='royalblue', label='Male')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_cols[order], fontsize=8)
        ax.set_xlabel('Contribution to importance (%)')
        ax.set_title(f'{model_name} SHAP importance contribution by sex (%)')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(sex_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved mean archetype SHAP importance by sex plot: {sex_path}')


def _safe_file_stem(value):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(value)).strip('_')


def _shap_all_classes_array(model_result):
    shap_values = model_result.get('shap_values_all_classes')
    if shap_values is None:
        return None

    try:
        shap_array = np.asarray(shap_values, dtype=float)
    except (TypeError, ValueError):
        try:
            shap_array = np.stack([np.asarray(v, dtype=float) for v in shap_values], axis=0)
        except (TypeError, ValueError):
            return None

    return shap_array if shap_array.ndim == 3 else None


def _archetype_title(label):
    text = str(label.item() if hasattr(label, 'item') else label)
    return text if text.lower().startswith('archetype') else f'Archetype {text}'


def _mann_whitney_pvalues_by_feature(values, feature_names, female_mask, male_mask):
    from scipy.stats import mannwhitneyu

    values = np.asarray(values, dtype=float)
    female_mask = np.asarray(female_mask, dtype=bool)
    male_mask = np.asarray(male_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        return {}

    pvalues = {}
    for feature_idx, feature_name in enumerate(feature_names):
        female_values = values[female_mask, feature_idx]
        male_values = values[male_mask, feature_idx]
        female_values = female_values[np.isfinite(female_values)]
        male_values = male_values[np.isfinite(male_values)]

        if len(female_values) == 0 or len(male_values) == 0:
            pvalues[feature_name] = np.nan
            continue

        try:
            _, pvalue = mannwhitneyu(female_values, male_values, alternative='two-sided')
        except ValueError:
            pvalue = np.nan
        pvalues[feature_name] = pvalue

    return pvalues


def _pvalue_to_asterisks(pvalue):
    if pd.isna(pvalue):
        return ''
    if pvalue <= 0.0001:
        return '****'
    if pvalue <= 0.001:
        return '***'
    if pvalue <= 0.01:
        return '**'
    if pvalue <= 0.05:
        return '*'
    return ''


def _annotate_sex_mann_whitney_stars(fig, axes, feature_pvalues):
    if not feature_pvalues:
        return

    fig.canvas.draw()
    reference_ax = None
    label_positions = []
    for ax in axes:
        labels = [(label.get_text(), tick) for label, tick in zip(ax.get_yticklabels(), ax.get_yticks())]
        labels = [(label, tick) for label, tick in labels if label]
        if labels:
            reference_ax = ax
            label_positions = labels
            break

    if reference_ax is None:
        return

    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    x_fig = (left_box.x1 + right_box.x0) / 2
    figure_inverse = fig.transFigure.inverted()

    for feature_name, y_data in label_positions:
        stars = _pvalue_to_asterisks(feature_pvalues.get(feature_name))
        if not stars:
            continue

        _, y_fig = figure_inverse.transform(reference_ax.transData.transform((0, y_data)))
        fig.text(x_fig, y_fig, stars, ha='center', va='center', fontsize=8,
                 fontweight='bold', color='black')


def plot_mean_archetype_shap_volcano_by_sex(loocv_results, source_df, directory_path, sex_col='sex'):
    try:
        import shap
    except ImportError:
        print('  Skipped SHAP volcano plots by sex; missing shap package')
        return

    output_dir = os.path.join(directory_path, 'mean_archetype_shap_volcano_by_sex')
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for model_name, model_result in loocv_results.items():
        metadata = model_result['metadata'].reset_index(drop=True)
        if sex_col not in metadata.columns:
            print(f'  Skipped SHAP volcano plots for {model_name}; missing column: {sex_col}')
            continue

        feature_cols = np.asarray(model_result['feature_cols'])
        missing_features = [feature for feature in feature_cols if feature not in source_df.columns]
        if missing_features:
            print(f'  Skipped SHAP volcano plots for {model_name}; missing features in source data')
            continue

        shap_all_classes = _shap_all_classes_array(model_result)
        if shap_all_classes is None:
            print(f'  Skipped SHAP volcano plots for {model_name}; class-specific SHAP values unavailable')
            continue

        n_features = len(feature_cols)
        if shap_all_classes.shape[1] != n_features and shap_all_classes.shape[2] == n_features:
            shap_all_classes = np.transpose(shap_all_classes, (0, 2, 1))
        if shap_all_classes.shape[1] != n_features:
            print(f'  Skipped SHAP volcano plots for {model_name}; SHAP feature shape mismatch')
            continue

        shap_pred_class = np.asarray(model_result['shap_values'], dtype=float)
        mean_abs_shap = np.nanmean(np.abs(shap_pred_class), axis=0)
        order = np.argsort(mean_abs_shap)[::-1]
        ordered_features = feature_cols[order].tolist()
        ordered_feature_values = source_df[ordered_features].reset_index(drop=True)

        sex_values = metadata[sex_col].astype(str).str.strip().str.lower()
        female_mask = sex_values.isin(['f', 'female']).to_numpy()
        male_mask = sex_values.isin(['m', 'male']).to_numpy()
        if not female_mask.any() and not male_mask.any():
            print(f'  Skipped SHAP volcano plots for {model_name}; no male/female labels')
            continue

        class_labels = np.unique(np.asarray(model_result.get('true_labels', [])))
        n_classes = shap_all_classes.shape[2]
        if len(class_labels) != n_classes:
            class_labels = np.arange(1, n_classes + 1)

        for class_idx, class_label in enumerate(class_labels):
            class_name = _archetype_title(class_label)
            class_shap = shap_all_classes[:, order, class_idx]
            feature_pvalues = _mann_whitney_pvalues_by_feature(
                class_shap, ordered_features, female_mask, male_mask)
            xlim = (-2, 2)

            fig, axes = plt.subplots(
                1, 2,
                figsize=(12, max(4.5, min(8, 0.22 * n_features + 1))),
                sharex=False,
                sharey=True,
                gridspec_kw={'wspace': 0.35},
            )

            for ax, mask, sex_name in zip(axes, [female_mask, male_mask], ['Female', 'Male']):
                if not mask.any():
                    ax.text(0.5, 0.5, f'No {sex_name.lower()} samples', transform=ax.transAxes,
                            ha='center', va='center')
                    ax.set_axis_off()
                    continue

                plt.sca(ax)
                shap.summary_plot(
                    class_shap[mask],
                    features=ordered_feature_values.loc[mask, ordered_features],
                    feature_names=ordered_features,
                    plot_type='violin',
                    max_display=n_features,
                    sort=False,
                    show=False,
                    color_bar=sex_name == 'Male',
                    color_bar_label='Feature value',
                    plot_size=None,
                )
                ax.axvline(0, color='gray', linewidth=0.6, alpha=0.7)
                ax.set_xlim(xlim)
                ax.set_title(f'{sex_name} (n={int(mask.sum())})', fontsize=8)
                ax.set_xlabel(ax.get_xlabel(), fontsize=7)
                ax.set_ylabel(ax.get_ylabel(), fontsize=7)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=5)

            for extra_ax in fig.axes[2:]:
                extra_ax.tick_params(labelsize=6)
                extra_ax.set_ylabel(extra_ax.get_ylabel(), fontsize=7)

            fig.suptitle(f'{model_name} {class_name} SHAP volcano by sex', fontsize=9)
            fig.subplots_adjust(left=0.28, right=0.94, bottom=0.12, top=0.90, wspace=0.35)
            _annotate_sex_mann_whitney_stars(fig, axes, feature_pvalues)

            file_name = (
                f'mean_archetype_shap_volcano_{_safe_file_stem(model_name)}_'
                f'{_safe_file_stem(class_name)}.pdf'
            )
            _disable_rasterization(fig)
            fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_count += 1

    if saved_count:
        print(f'  Saved {saved_count} mean archetype SHAP volcano by sex plots: {output_dir}')


def run_parallel_iterations(directory_path, all_pca_coords, no_pca_data, exclude_pca_data,
                            behavior_df, metadata_cols, hormones_df, sample_frac,
                            user_models, max_iterations, n_parallel_workers,
                            min_per_vertex, n_permutations):
    state = setup_iteration_state()
    accepted_results = []
    mean_coords_df = None
    pending = {}
    # creates a parallel process pool
    executor = ProcessPoolExecutor(
        max_workers=n_parallel_workers,
        initializer=_init_iteration_worker,
        initargs=(
            all_pca_coords, sample_frac, no_pca_data, behavior_df, metadata_cols,
            hormones_df, state['if_dominant_archetype'], user_models,
            min_per_vertex, n_permutations,
        ),
    )

    print(f'Running up to {n_parallel_workers} iterations in parallel')
    try:
        while state['iteration'] < max_iterations:
            while len(pending) < n_parallel_workers:
                state['attempts'] += 1
                # future is the work
                future = executor.submit(_run_iteration_candidate, state['attempts'])
                pending[future] = state['attempts']

            done, _ = wait(pending, return_when=FIRST_COMPLETED)  # wait until at least one future is done
            for future in done:  # in done are the futures that are completed
                pending.pop(future)  # remove the completed future from pending
                result = future.result()

                if result['idx_tuple'] in state['seen_indices']:
                    print(f'  Skipped duplicate sample (attempt {result["attempt"]})')
                    continue
                state['seen_indices'].add(result['idx_tuple'])

                if not result['accepted']:
                    print(f'  Rejected (attempt {result["attempt"]}) — counts per vertex: {result["counts"]}')
                    continue

                state['iteration'] += 1
                result['iteration'] = state['iteration']

                if state['reference_archetypes'] is None:
                    state['reference_archetypes'] = result['archetypes'].copy()

                mapping = get_alignment_mapping(state['reference_archetypes'], result['archetypes'])
                _apply_reference_mapping(result, mapping)
                accepted_results.append(result)

                for metric in result['metrics']:
                    metric['iteration'] = state['iteration']
                state['iteration_metrics'].extend(result['metrics'])

                state['all_true'], state['all_preds_by_model'] = accumulate_results(
                    state['all_true'], state['all_preds_by_model'],
                    result['aligned_true'], result['loocv_results'])
                state['last_loocv'] = result['loocv_results']
                state['last_aligned_true'] = result['aligned_true']

                print(f'Iteration {state["iteration"]}/{max_iterations} accepted — '
                      f'Var explained: {result["varexlp"]:.3f}  '
                      f'Counts per vertex: {result["counts"]}')
                mapping_text = ', '.join(
                    f'{current_idx + 1}->{ref_idx + 1}'
                    for current_idx, ref_idx in enumerate(result['mapping']))
                print(f'  Archetype mapping current->reference: {mapping_text}')
                for mname, mres in result['loocv_results'].items():
                    pvalue = result['pvalues'].get(mname)
                    print(f'  {mname} LOOCV accuracy: {mres["accuracy"]:.3f}  p={pvalue:.4f}')

                _save_iteration_pdf(directory_path, all_pca_coords, no_pca_data,
                                    exclude_pca_data, user_models, result)

                if state['iteration'] >= max_iterations:
                    break
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    if accepted_results:
        _save_parallel_iteration_outputs(directory_path, accepted_results)
        save_classification_metrics(state['iteration_metrics'], state['all_true'],
                                    state['all_preds_by_model'], directory_path)
        _plot_permutation_mean_vs_real_fscore(directory_path, accepted_results, user_models)
        mean_coords_df, ellipse_geometry_df = plot_mean_archetype_triangle(all_pca_coords, accepted_results, os.path.join(directory_path, "mean_archetype_triangle.pdf"))
        with pd.ExcelWriter(os.path.join(directory_path, "mean_archetype_coordinates.xlsx")) as writer:
            mean_coords_df.to_excel(writer, sheet_name="mean_coordinates", index=False)
            ellipse_geometry_df.to_excel(writer, sheet_name="ellipse_geometry", index=False)
        plot_average_confusion_matrices(accepted_results, os.path.join(directory_path, "average_confusion_matrices.pdf"), model_names=user_models)

    return accepted_results, state, mean_coords_df


def run_mean_archetype_analysis(directory_path, mean_coords_df, behavior_df, metadata_cols,
                                all_pca_coords, hormones_df, if_dominant_archetype,
                                user_models, n_permutations):
    mean_output_dir = os.path.join(directory_path, 'Data_of_mean_triangle')
    os.makedirs(mean_output_dir, exist_ok=True)
    mean_coords_output_path = os.path.join(mean_output_dir, "mean_archetype_coordinates.xlsx")
    mean_coords_source_path = os.path.join(directory_path, "mean_archetype_coordinates.xlsx")

    if mean_coords_df is None:
        mean_coords_path = (mean_coords_output_path if os.path.exists(mean_coords_output_path)
                            else mean_coords_source_path)
        if not os.path.exists(mean_coords_path):
            print(f'Skipping mean archetype analysis; missing file: {mean_coords_path}')
            return None, None, None, None
        mean_coords_df = pd.read_excel(mean_coords_path)
        print(f'Loaded mean archetype coordinates: {mean_coords_path}')

    if not os.path.exists(mean_coords_output_path):
        if os.path.exists(mean_coords_source_path):
            shutil.copy2(mean_coords_source_path, mean_coords_output_path)
            print(f'  Saved mean archetype coordinates in mean triangle folder: {mean_coords_output_path}')
        else:
            with pd.ExcelWriter(mean_coords_output_path, engine='openpyxl') as writer:
                mean_coords_df.to_excel(writer, sheet_name="mean_coordinates", index=False)
            print(f'  Saved mean archetype coordinates in mean triangle folder: {mean_coords_output_path}')

    mean_table_df, mean_hormones_arch = build_mean_archetype_assignment(
        mean_coords_df=mean_coords_df,
        behavior_df=behavior_df,
        metadata_cols=metadata_cols,
        all_pca_coords=all_pca_coords,
        hormones_df=hormones_df,
        if_dominant_archetype=if_dominant_archetype,
        directory_path=mean_output_dir,
    )
    plot_mean_hormone_archetype_pc1_pc2_by_sex(
        mean_hormones_arch, mean_output_dir, mean_coords_df=mean_coords_df)
    plot_mean_hormone_feature_correlations_with_archetype_probs(
        mean_hormones_arch, metadata_cols, mean_output_dir)
    plot_mean_behavior_feature_correlations_with_mean_pcs(
        behavior_df, mean_table_df, metadata_cols, mean_output_dir)
    
    loocv_results = predict_archetype_loocv_with_shap(
        mean_hormones_arch, metadata_cols, model_names=user_models)
    
    save_mean_archetype_shap_values(loocv_results, mean_hormones_arch, mean_output_dir)
    plot_mean_archetype_shap_importance(loocv_results, mean_output_dir)
    plot_mean_archetype_shap_volcano_by_sex(loocv_results, mean_hormones_arch, mean_output_dir)

    permutation_results = calculate_loocv_permutation_pvalues(
        mean_hormones_arch,
        metadata_cols,
        loocv_results,
        mean_hormones_arch['Dominant_archetype'].values,
        model_names=user_models,
        n_permutations=n_permutations,
    )
    plot_loocv_confusion_matrices(
        loocv_results,
        mean_hormones_arch['Dominant_archetype'].values,
        os.path.join(mean_output_dir, 'mean_hormones_loocv_confusion_matrix.pdf'),
        permutation_results=permutation_results,
    )
    
    return mean_table_df, mean_hormones_arch, loocv_results, permutation_results


def save_mean_archetype_behavior_correlation_data(directory_path, mean_coords_df,
                                                  behavior_df, metadata_cols,
                                                  all_pca_coords):
    if mean_coords_df is None:
        mean_coords_path = os.path.join(directory_path, "mean_archetype_coordinates.xlsx")
        if not os.path.exists(mean_coords_path):
            print(f'Skipping correlation data; missing file: {mean_coords_path}')
            return None
        mean_coords_df = pd.read_excel(mean_coords_path)
        print(f'Loaded mean archetype coordinates: {mean_coords_path}')


    mean_archetypes = mean_coords_df[['PC1', 'PC2']].to_numpy(dtype=float)
    prob_coeffs = compute_archetype_probabilities(all_pca_coords, mean_archetypes)
    prob_table_df = build_table_df(behavior_df, metadata_cols, all_pca_coords, prob_coeffs)
    prob_cols = ['PC1', 'PC2', 'Archetype1_prob', 'Archetype2_prob',
                 'Archetype3_prob', 'Dominant_archetype']
    correlation_df = pd.concat(
        [behavior_df.reset_index(drop=True), prob_table_df[prob_cols].reset_index(drop=True)],
        axis=1,
    )

    output_path = os.path.join(directory_path, 'mean_archetype_behavior_for_correlation.xlsx')
    correlation_df.to_excel(output_path, index=False)
    print(f'  Saved mean archetype behavior correlation data: {output_path}')
    return correlation_df


def _bh_correction(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(pvalues)

    if not valid_mask.any():
        return adjusted

    _, adjusted_pvalues, _, _ = multipletests(
        pvalues[valid_mask],
        alpha=0.05,
        method='fdr_bh',
    )

    adjusted[valid_mask] = adjusted_pvalues
    return adjusted


def save_archetype_behavior_correlations(correlation_df, metadata_cols, directory_path,
                                         pvalue_threshold=0.05):
    from scipy.stats import pearsonr

    target_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    if correlation_df is None:
        print('Skipping archetype behaviour correlations; missing correlation dataframe')
        return None

    missing_target_cols = [col for col in target_cols if col not in correlation_df.columns]
    if missing_target_cols:
        print(f'Skipping archetype behaviour correlations; missing columns: {missing_target_cols}')
        return None

    excluded_cols = set(metadata_cols + [
        'PC1', 'PC2', 'Dominant_archetype',
        'Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob',
    ])
    behavior_cols = [
        col for col in correlation_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(correlation_df[col])
    ]

    rows = []
    for arch_idx, target_col in enumerate(target_cols, start=1):
        for col in behavior_cols:
            pair_df = correlation_df[[target_col, col]].dropna()
            if len(pair_df) < 3 or pair_df[target_col].nunique() < 2 or pair_df[col].nunique() < 2:
                r_value = np.nan
                p_value = np.nan
            else:
                r_value, p_value = pearsonr(pair_df[target_col], pair_df[col])

            rows.append({
                'Archetype': f'Archetype{arch_idx}',
                'Behavior': col,
                'Pearson_r': r_value,
                'p_value': p_value,
                'n': len(pair_df),
            })

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        print('Skipping archetype behaviour correlations; no numeric behaviour columns found')
        return corr_df

    #BH correction is done separately for each archetype to avoid mixing p-values across different archetypes
    corr_df['p_adjusted_BH'] = np.nan
    for archetype in corr_df['Archetype'].unique():
        mask = corr_df['Archetype'] == archetype
        corr_df.loc[mask, 'p_adjusted_BH'] = _bh_correction(corr_df.loc[mask, 'p_value'].values)

    corr_df = corr_df.sort_values(['Archetype', 'p_adjusted_BH'], na_position='last')
    significant_df = corr_df[corr_df['p_adjusted_BH'] < pvalue_threshold].copy()

    output_path = os.path.join(directory_path, 'archetype_behavior_correlations_BH_p_lt_0_05.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='all_correlations', index=False)
        significant_df.to_excel(writer, sheet_name='BH_p_lt_0_05', index=False)

    plot_path = os.path.join(directory_path, 'archetype_behavior_correlations_BH_p_lt_0_05.pdf')
    plot_significant_archetype_behavior_correlations(significant_df, plot_path, pvalue_threshold)

    print(f'  Saved archetype behaviour correlations: {output_path}')
    print(f'  Significant BH-adjusted correlations p < {pvalue_threshold}: {len(significant_df)}')
    return significant_df


def plot_significant_archetype_behavior_correlations(significant_df, output_path,
                                                     pvalue_threshold=0.05):
    archetypes = ['Archetype1', 'Archetype2', 'Archetype3']
    max_rows = 1
    if significant_df is not None and not significant_df.empty:
        max_rows = max(max_rows, *(len(significant_df[significant_df['Archetype'] == arch])
                                  for arch in archetypes))

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.35 * max_rows)), sharex=True)
    colors = ['tomato', 'seagreen', 'royalblue']

    for ax, archetype, color in zip(axes, archetypes, colors):
        if significant_df is None or significant_df.empty:
            arch_df = pd.DataFrame()
        else:
            arch_df = significant_df[significant_df['Archetype'] == archetype].copy()
            arch_df = arch_df.sort_values('Pearson_r', ascending=True)

        if arch_df.empty:
            ax.text(0.5, 0.5, 'No significant\ncorrelations', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_yticks([])
        else:
            ax.barh(arch_df['Behavior'], arch_df['Pearson_r'], color=color, alpha=0.75)
            ax.tick_params(axis='y', labelsize=10)

        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(archetype)
        ax.set_xlabel('Pearson r')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle(f'BH-significant behaviour correlations (p < {pvalue_threshold})')
    fig.tight_layout()
    _disable_rasterization(fig)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved archetype behaviour correlation plot: {output_path}')


# input: None (uses hardcoded directory_path)
# output: None (runs 50-iteration archetype analysis + LOOCV + SHAP pipeline)
def main():
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Dana_model_07_07_2026'
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal', 'Days']
    max_iterations = 50
    n_parallel_workers = max(1, (os.cpu_count() or 1) - 1)
    n_permutations = 200
    min_per_vertex = 40
    select_days = False
    days_used = range(1, 7)
    exclude_pca_data = False
    sample_frac=0.8
    user_models = ['XGBoost'] #['LogReg', 'SVM', 'MLP', 'RandomForest', 'ExtraTrees', 'HistGB', 'XGBoost']
    run_parallel_iteration_analysis = False
    run_mean_archetype_analysis_flag = True
    run_correlation = False
    n_permutations_for_mean_archetype_analysis = 200
    behavior_df, hormones_df = load_data(directory_path)



    if select_days:
        behavior_df = behavior_df[behavior_df['Days'].isin(days_used)]

    pca_model, all_pca_coords, behavior_cols, pca_metadata_df = compute_pca(behavior_df)
    pca_metadata_df.to_excel(os.path.join(directory_path, 'pca_coordinates_with_metadata.xlsx'), index=False)

    if  exclude_pca_data:
        no_pca_data = pd.read_excel(os.path.join(directory_path, 'Data_pca_to_remove.xlsx'))
        # coords_to_remove = no_pca_data[['PC1','PC2']].to_numpy(dtype=float)
        # matches = np.isclose(all_pca_coords[:, None, :],coords_to_remove[None, :, :]).all(axis=2).any(axis=1)
        # pca_coords_filtered = all_pca_coords[~matches]
        # all_pca_coords = pca_coords_filtered
        # #also remove the rows from behavior_df
        # behavior_df = behavior_df.loc[~matches].reset_index(drop=True)
    else:
        no_pca_data = None


    save_behavior_parameters(behavior_cols, directory_path)

    accepted_results = []
    state = None
    mean_coords_df = None
    if run_parallel_iteration_analysis:
        accepted_results, state, mean_coords_df = run_parallel_iterations(
            directory_path, all_pca_coords, no_pca_data, exclude_pca_data,
            behavior_df, metadata_cols, hormones_df, sample_frac, user_models,
            max_iterations, n_parallel_workers, min_per_vertex, n_permutations)
    else:
        print('Skipping parallel iteration analysis')

    if run_mean_archetype_analysis_flag:
        if_dominant_archetype = state['if_dominant_archetype'] if state is not None else False
        run_mean_archetype_analysis(
            directory_path, mean_coords_df, behavior_df, metadata_cols,
            all_pca_coords, hormones_df, if_dominant_archetype, user_models,
            n_permutations_for_mean_archetype_analysis,
        )
    else:
        print('Skipping mean archetype analysis')

    if run_correlation:
        data_for_correlation = save_mean_archetype_behavior_correlation_data(
            directory_path, mean_coords_df, behavior_df, metadata_cols, all_pca_coords,
        )
        save_archetype_behavior_correlations(data_for_correlation, metadata_cols, directory_path)
    else:
        print('Skipping mean archetype behavior correlation data')




if __name__ == "__main__":
    main()
