import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from archetype_analysis import compute_pca, sample_and_fit_archetypes, apply_pca_transform, compute_archetype_probabilities, assign_to_nearest_vertex, predict_archetype_loocv
import os
import openpyxl



def main():
    #=========
    #User input
    #====================
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use'
    behavior_every_day_file = 'Behavior_every_day.xlsx'
    behavior_mean_file = 'Behavior_mean_per_day.xlsx'
    hormones = 'Hormones.xlsx'
    #=========
    # Load the data 
    behavior_every_day_df = pd.read_excel(f'{directory_path}/{behavior_every_day_file}')
    behavior_mean_df = pd.read_excel(f'{directory_path}/{behavior_mean_file}')
    hormones_df = pd.read_excel(f'{directory_path}/{hormones}')
    #=========
    min_per_vertex = 40  #with all the data
    #do pca on all data to get the coordinates for the archetype fitting
    pca_model, all_pca_coords, behavior_cols = compute_pca(behavior_every_day_df)

    # Apply PCA to behavior_mean_df before the loop
    mean_pca_coords = apply_pca_transform(pca_model, behavior_mean_df, behavior_cols)
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']

    plt.ion()
    fig = plt.figure(figsize=(22, 6))
    gs = GridSpec(1, 4, figure=fig)
    ax_tri = fig.add_subplot(gs[0, 0])
    cm_axes = {
        'LogReg': fig.add_subplot(gs[0, 1]),
        'MLP': fig.add_subplot(gs[0, 2]),
        'XGBoost': fig.add_subplot(gs[0, 3]),
    }

    iteration = 0
    attempts = 0
    tables = []

    while iteration < 50:
        attempts += 1
        pca_coords, archetypes, varexlp, counts = sample_and_fit_archetypes(all_pca_coords)

        if all(c >= min_per_vertex for c in counts):
            iteration += 1
            ax_tri.clear()
            ax_tri.scatter(all_pca_coords[:, 0], all_pca_coords[:, 1], alpha=0.4, s=20, c='steelblue', label='Every day')
            ax_tri.scatter(mean_pca_coords[:, 0], mean_pca_coords[:, 1], alpha=0.7, s=40, c='orange', marker='s', label='Mean')

            tri_x = [archetypes[0, 0], archetypes[1, 0], archetypes[2, 0], archetypes[0, 0]]
            tri_y = [archetypes[0, 1], archetypes[1, 1], archetypes[2, 1], archetypes[0, 1]]
            ax_tri.plot(tri_x, tri_y, 'k-', linewidth=2, label='Archetype triangle')
            ax_tri.scatter(archetypes[:, 0], archetypes[:, 1], c='r', s=100, marker='^', zorder=5)

            mean_counts = assign_to_nearest_vertex(mean_pca_coords, archetypes)
            for v in range(3):
                ax_tri.annotate(f'every: {counts[v]}  mean: {mean_counts[v]}',
                                archetypes[v], textcoords='offset points',
                                xytext=(0, 12), ha='center', fontsize=8, color='red')

            ax_tri.set_xlabel('PC1')
            ax_tri.set_ylabel('PC2')
            ax_tri.set_title(f'Iteration {iteration}/50 - Var explained: {varexlp:.3f}   '
                             f'(attempts: {attempts})')
            ax_tri.set_aspect('equal')
            ax_tri.margins(0.1)
            ax_tri.legend(fontsize=7)
            ax_tri.grid(True, alpha=0.3)

            print(f'Iteration {iteration}/50 accepted - Var explained: {varexlp:.3f}  '
                  f'Counts per vertex: {counts}')

            # Compute probability table for this iteration (in memory)
            prob_coeffs = compute_archetype_probabilities(mean_pca_coords, archetypes)
            table_df = behavior_mean_df[metadata_cols].copy()
            table_df['PC1'] = mean_pca_coords[:, 0]
            table_df['PC2'] = mean_pca_coords[:, 1]
            table_df['Archetype1_prob'] = prob_coeffs[:, 0]
            table_df['Archetype2_prob'] = prob_coeffs[:, 1]
            table_df['Archetype3_prob'] = prob_coeffs[:, 2]
            prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
            table_df['Dominant_archetype'] = table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
            tables.append(table_df)
            # Merge dominant archetype with hormones
            merge_cols = list(set(behavior_mean_df.columns) & set(hormones_df.columns))
            hormones_arch = hormones_df.merge(table_df[merge_cols + ['Dominant_archetype']], on=merge_cols, how='left')
            # LOOCV prediction
            loocv_results = predict_archetype_loocv(hormones_arch, metadata_cols)
            for mname, mres in loocv_results.items():
                print(f'  {mname} LOOCV accuracy: {mres["accuracy"]:.3f}')
            # Plot confusion matrices
            for mname, mres in loocv_results.items():
                ax = cm_axes[mname]
                ax.clear()
                f1_macro = f1_score(hormones_arch['Dominant_archetype'], mres['predictions'], average='macro')
                ConfusionMatrixDisplay.from_predictions(
                    hormones_arch['Dominant_archetype'], mres['predictions'],
                    display_labels=[1, 2, 3], ax=ax, colorbar=False,
                    cmap='Blues', text_kw={'fontsize': 8},
                )
                ax.set_title(f'{mname}  acc={mres["accuracy"]:.3f}  F1={f1_macro:.3f}', fontsize=10)
            plt.pause(0.8)

        else:
            print(f'  Rejected (attempt {attempts}) — counts per vertex: {counts}')

    plt.ioff()
    plt.show()

    
    




if __name__ == "__main__":
    main()