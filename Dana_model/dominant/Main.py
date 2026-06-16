import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from archetype_analysis import compute_pca, sample_and_fit_archetypes, compute_archetype_probabilities, predict_archetype_loocv, select_top_per_archetype
from alignment.label_alignment import get_alignment_mapping, apply_mapping, accumulate_results, compute_aggregate_confusion
from plot_utils import setup_figure, draw_triangle, plot_confusion_matrices, save_if_best, plot_aggregate_confusion, plot_permutation_tests
import os
import openpyxl


def main():
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use'
    behavior_every_day_file = 'Behavior_every_day.xlsx'
    hormones = 'Hormones.xlsx'

    behavior_every_day_df = pd.read_excel(f'{directory_path}/{behavior_every_day_file}')
    hormones_df = pd.read_excel(f'{directory_path}/{hormones}')

    min_per_vertex = 40
    pca_model, all_pca_coords, behavior_cols = compute_pca(behavior_every_day_df)
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']

    plt.ion()
    fig, ax_tri, cm_axes, perm_axes = setup_figure()

    iteration = 0
    attempts = 0
    tables = []
    best_f1 = 0.48
    best_iter = 0
    reference_archetypes = None
    all_true = []
    all_preds_by_model = []
    seen_indices = set()

    while iteration < 100:
        attempts += 1
        pca_coords, archetypes, varexlp, counts, sample_indices = sample_and_fit_archetypes(all_pca_coords)
        idx_tuple = tuple(sorted(sample_indices))
        if idx_tuple in seen_indices:
            continue
        seen_indices.add(idx_tuple)

        if all(c >= min_per_vertex for c in counts):
            iteration += 1
            if reference_archetypes is None:
                reference_archetypes = archetypes.copy()
            mapping = get_alignment_mapping(reference_archetypes, archetypes)
            draw_triangle(ax_tri, all_pca_coords, archetypes, counts, varexlp, iteration, attempts,
                          max_iterations=100, mapping=mapping)

            print(f'Iteration {iteration}/100 accepted - Var explained: {varexlp:.3f}  '
                  f'Counts per vertex: {counts}')

            prob_coeffs = compute_archetype_probabilities(all_pca_coords, archetypes)
            table_df = behavior_every_day_df[metadata_cols].copy()
            table_df['PC1'] = all_pca_coords[:, 0]
            table_df['PC2'] = all_pca_coords[:, 1]
            table_df['Archetype1_prob'] = prob_coeffs[:, 0]
            table_df['Archetype2_prob'] = prob_coeffs[:, 1]
            table_df['Archetype3_prob'] = prob_coeffs[:, 2]
            prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
            table_df['Dominant_archetype'] = table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
            tables.append(table_df)

            top_df = select_top_per_archetype(table_df, archetypes)

            hormones_arch = top_df.merge(
                hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
                on=['Experiment', 'sex', 'Hierarchy'], how='inner'
            )
            drop_cols = ['n_days', 'cum_arch1', 'cum_arch2', 'cum_arch3']
            hormones_arch = hormones_arch.drop(columns=[c for c in drop_cols if c in hormones_arch.columns])

            loocv_results = predict_archetype_loocv(hormones_arch, metadata_cols)
            for mname, mres in loocv_results.items():
                mres['predictions'] = apply_mapping(mres['predictions'], mapping)
            aligned_true = apply_mapping(hormones_arch['Dominant_archetype'].values, mapping)
            all_true, all_preds_by_model = accumulate_results(all_true, all_preds_by_model, aligned_true, loocv_results)

            for mname, mres in loocv_results.items():
                print(f'  {mname} LOOCV accuracy: {mres["accuracy"]:.3f}')

            f1_scores, per_class_f1 = plot_confusion_matrices(cm_axes, loocv_results, hormones_arch['Dominant_archetype'])

            plot_permutation_tests(perm_axes, loocv_results, hormones_arch['Dominant_archetype'])

            saved = save_if_best(fig, directory_path, f1_scores, per_class_f1, iteration)
            if saved:
                best_iter = iteration
            plt.pause(0.8)
        else:
            print(f'  Rejected (attempt {attempts}) — counts per vertex: {counts}')

    agg_confusion = compute_aggregate_confusion(all_true, all_preds_by_model)
    plot_aggregate_confusion(agg_confusion, directory_path)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
