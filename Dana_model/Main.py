import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from archetype_analysis import (compute_pca, predict_archetype_loocv, sample_and_fit_archetypes,
    compute_archetype_probabilities, build_mean_archetype_assignment)
from alignment.label_alignment import accumulate_results, apply_mapping
from plot_utils import (plot_mean_archetype_triangle, plot_average_confusion_matrices,
                        setup_figure, draw_triangle, plot_confusion_matrices)
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

from main_utils import (
    load_data, save_behavior_parameters, setup_iteration_state,
    build_table_df, select_top_and_merge_hormones,
    predict_and_align, compute_iteration_metrics, save_classification_metrics,
)


_WORKER_CONTEXT = {}


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
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved permutation mean vs real F-score plot: {output_path}')


# input: None (uses hardcoded directory_path)
# output: None (runs 50-iteration archetype analysis + LOOCV + SHAP pipeline)
def main():
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use\outlier'
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal', 'Days']
    max_iterations = 50
    n_parallel_workers = max(1, (os.cpu_count() or 1) - 1)
    n_permutations = 4
    min_per_vertex = 40
    select_days = False
    days_used = range(1, 7)
    exclude_pca_data = False
    sample_frac=0.8
    user_models = ['XGBoost'] #['LogReg', 'SVM', 'MLP', 'RandomForest', 'ExtraTrees', 'HistGB', 'XGBoost']

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

    state = setup_iteration_state()
    accepted_results = []
    pending = {}
    #creates a parallel process pool
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
                #future is the works
                future = executor.submit(_run_iteration_candidate, state['attempts'])
                pending[future] = state['attempts']

            done, _ = wait(pending, return_when=FIRST_COMPLETED) #wait until at least one future is done
            for future in done: #in done are the futures that are completed
                pending.pop(future) #remove the completed future from pending
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
        mean_coords_df = plot_mean_archetype_triangle(all_pca_coords, accepted_results, os.path.join(directory_path, "mean_archetype_triangle.pdf"))
        mean_coords_df.to_excel(os.path.join(directory_path, "mean_archetype_coordinates.xlsx"),index=False)
        plot_average_confusion_matrices(accepted_results,os.path.join(directory_path, "average_confusion_matrices.pdf"),model_names=user_models)
        mean_table_df, mean_hormones_arch = build_mean_archetype_assignment(mean_coords_df=mean_coords_df, behavior_df=behavior_df, metadata_cols=metadata_cols,
            all_pca_coords=all_pca_coords, hormones_df=hormones_df, if_dominant_archetype=state['if_dominant_archetype'], directory_path=directory_path)
        loocv_results = predict_archetype_loocv(mean_hormones_arch, metadata_cols, model_names= user_models)




if __name__ == "__main__":
    main()
