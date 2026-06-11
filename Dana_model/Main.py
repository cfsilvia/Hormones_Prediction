import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from archetype_analysis import compute_pca, sample_and_fit_archetypes, apply_pca_transform, compute_archetype_probabilities
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
    fig, ax = plt.subplots(figsize=(8, 6))

    iteration = 0
    attempts = 0
    tables = []

    while iteration < 50:
        attempts += 1
        pca_coords, archetypes, varexlp, counts = sample_and_fit_archetypes(all_pca_coords)

        if all(c >= min_per_vertex for c in counts):
            iteration += 1
            ax.clear()
            ax.scatter(all_pca_coords[:, 0], all_pca_coords[:, 1], alpha=0.4, s=20, c='steelblue', label='Every day')
            ax.scatter(mean_pca_coords[:, 0], mean_pca_coords[:, 1], alpha=0.7, s=40, c='orange', marker='s', label='Mean')

            tri_x = [archetypes[0, 0], archetypes[1, 0], archetypes[2, 0], archetypes[0, 0]]
            tri_y = [archetypes[0, 1], archetypes[1, 1], archetypes[2, 1], archetypes[0, 1]]
            ax.plot(tri_x, tri_y, 'k-', linewidth=2, label='Archetype triangle')

            ax.scatter(archetypes[:, 0], archetypes[:, 1], c='r', s=100, marker='^', zorder=5)

            for v in range(3):
                ax.annotate(f'n={counts[v]}', archetypes[v], textcoords='offset points',
                            xytext=(0, 12), ha='center', fontsize=9, color='red')

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'Iteration {iteration}/50 - Var explained: {varexlp:.3f}   '
                         f'(attempts: {attempts})')
            ax.set_aspect('equal')
            ax.margins(0.1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.pause(0.5)
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
            tables.append(table_df)
        else:
            print(f'  Rejected (attempt {attempts}) — counts per vertex: {counts}')

    plt.ioff()
    plt.show()

    
    




if __name__ == "__main__":
    main()