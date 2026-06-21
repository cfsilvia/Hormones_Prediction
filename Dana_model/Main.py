import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from archetype_analysis import (compute_pca, sample_and_fit_archetypes,
    compute_archetype_probabilities)
from alignment.label_alignment import get_alignment_mapping, accumulate_results
from plot_utils import (setup_figure, draw_triangle, plot_confusion_matrices, save_if_best,
    plot_aggregate_confusion, plot_permutation_tests, plot_aggregate_permutation,
    plot_permutation_per_archetype, plot_shap_per_archetype)
from shap_calculation import train_full_models_and_shap
from sklearn.preprocessing import StandardScaler
import os

from main_utils import (
    load_data, save_behavior_parameters, setup_iteration_state,
    build_table_df, select_top_and_merge_hormones,
    predict_and_align, compute_iteration_metrics, get_feature_cols,
    save_classification_metrics, save_aggregated_shap, compute_aggregate_metrics,
)


# input: None (uses hardcoded directory_path)
# output: None (runs 50-iteration archetype analysis + LOOCV + SHAP pipeline)
def main():
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use'
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']
    min_per_vertex = 40

    behavior_df, hormones_df = load_data(directory_path)

    pca_model, all_pca_coords, behavior_cols = compute_pca(behavior_df)
    save_behavior_parameters(behavior_cols, directory_path)

    plt.ion()
    fig, ax_tri, cm_axes, perm_axes = setup_figure()
    state = setup_iteration_state()

    while state['iteration'] < 50:
        state['attempts'] += 1
        pca_coords, archetypes, varexlp, counts, sample_indices = sample_and_fit_archetypes(all_pca_coords)

        # Avoid re-evaluating the same sampled data points multiple times
        idx_tuple = tuple(sorted(sample_indices))
        if idx_tuple in state['seen_indices']:
            continue #break the loop
        state['seen_indices'].add(idx_tuple)

        if all(c >= min_per_vertex for c in counts):
            state['iteration'] += 1
            if state['reference_archetypes'] is None:
                state['reference_archetypes'] = archetypes.copy()
            #compare with reference triangle and get mapping for label alignment    
            mapping = get_alignment_mapping(state['reference_archetypes'], archetypes)
            draw_triangle(ax_tri, all_pca_coords, archetypes, counts, varexlp,
                          state['iteration'], state['attempts'], mapping=mapping)
            print(f'Iteration {state["iteration"]}/50 accepted — Var explained: {varexlp:.3f}  '
                  f'Counts per vertex: {counts}')

            prob_coeffs = compute_archetype_probabilities(all_pca_coords, archetypes)
            table_df = build_table_df(behavior_df, metadata_cols, all_pca_coords, prob_coeffs)

            hormones_arch = select_top_and_merge_hormones(
                table_df, archetypes, hormones_df, state['if_dominant_archetype'])

            loocv_results, aligned_true = predict_and_align(
                hormones_arch, metadata_cols, state['if_dominant_archetype'], mapping)
            
            #Accumulate
            state['all_true'], state['all_preds_by_model'] = accumulate_results(
                state['all_true'], state['all_preds_by_model'], aligned_true, loocv_results)

            metrics = compute_iteration_metrics(aligned_true, loocv_results)
            for m in metrics:
                m['iteration'] = state['iteration']
            state['iteration_metrics'].extend(metrics) #add to the list of dicts for all iterations 

            for mname, mres in loocv_results.items():
                print(f'  {mname} LOOCV accuracy: {mres["accuracy"]:.3f}')

            f1_scores, per_class_f1 = plot_confusion_matrices(cm_axes, loocv_results, aligned_true)
            pvalues = plot_permutation_tests(perm_axes, loocv_results, aligned_true)
            state['last_loocv'] = loocv_results
            state['last_aligned_true'] = aligned_true

            #shap
            shap_results = train_full_models_and_shap(hormones_arch, metadata_cols, aligned_true)
            if state['feature_cols'] is None:
                state['feature_cols'] = get_feature_cols(hormones_arch, metadata_cols)

            X_full = hormones_arch[state['feature_cols']].select_dtypes(include=[np.number]).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)

            for m in shap_results:
                shap_vals = shap_results[m]
                if isinstance(shap_vals, list):
                    shap_vals = np.array(shap_vals).transpose(1, 2, 0)
                per_feature_mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
                if m not in state['aggregated_shap']:
                    state['aggregated_shap'][m] = {'mean_abs': [], 'per_class_mean_abs': [],
                                                   'p_values': [], 'raw_shap': [], 'raw_features': []}
                state['aggregated_shap'][m]['mean_abs'].append(per_feature_mean_abs)
                state['aggregated_shap'][m]['per_class_mean_abs'].append(
                    np.mean(np.abs(shap_vals), axis=0))
                state['aggregated_shap'][m]['p_values'].append(pvalues[m])
                state['aggregated_shap'][m]['raw_shap'].append(shap_vals)
                state['aggregated_shap'][m]['raw_features'].append(
                    X_full if m == 'XGBoost' else X_scaled)
            state['shap_iter_count'] += 1
            ###################################################
            save_if_best(fig, directory_path, f1_scores, per_class_f1, state['iteration'])
            plt.pause(0.8)
        else:
            print(f'  Rejected (attempt {state["attempts"]}) — counts per vertex: {counts}')

    agg_confusion = compute_aggregate_metrics(state['all_true'], state['all_preds_by_model'])
    plot_aggregate_confusion(agg_confusion, directory_path)
    plot_aggregate_permutation(state['all_true'], state['all_preds_by_model'], directory_path)

    if state['all_true'] and state['all_preds_by_model']:
        plot_permutation_per_archetype(state['all_true'], state['all_preds_by_model'], directory_path)

    save_classification_metrics(state['iteration_metrics'], state['all_true'],
                                 state['all_preds_by_model'], directory_path)
    save_aggregated_shap(state['aggregated_shap'], state['feature_cols'],
                         state['shap_iter_count'], directory_path)

    if state['aggregated_shap']:
        plot_shap_per_archetype(state['aggregated_shap'], state['feature_cols'], directory_path)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
