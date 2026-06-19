import pandas as pd
import numpy as np
from archetype_analysis import compute_pca, compute_archetype_probabilities, assign_to_nearest_vertex
from py_pcha import PCHA
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
import os


def _normalize_column_names(df):
    df = df.copy()
    df.columns = [c.replace(' ', '.') for c in df.columns]
    return df


def run_pca(data_df):
    return compute_pca(data_df)


def find_best_triangle(pca_coords, n_trials=500):
    best_archetypes = None
    best_varexlp = -np.inf
    best_counts = None

    n = pca_coords.shape[0]
    n_sample = max(int(n * 0.8), 10)
    for _ in range(n_trials):
        sample_indices = np.random.choice(n, n_sample, replace=False)
        sampled = pca_coords[sample_indices]
        try:
            XC, S, C, SSE, varexlp = PCHA(sampled.T, 3)
        except Exception:
            continue
        archetypes = np.array(XC.T)
        counts = assign_to_nearest_vertex(pca_coords, archetypes)
        if all(c >= 2 for c in counts) and varexlp > best_varexlp:
            best_varexlp = varexlp
            best_archetypes = archetypes
            best_counts = counts

    return best_archetypes, best_varexlp, best_counts


def _bh_correction(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    ranked = np.argsort(pvals)
    sorted_p = pvals[ranked]
    bh_threshold = np.minimum(sorted_p * n / (np.arange(1, n + 1)), 1.0)
    rejected = sorted_p <= bh_threshold
    p_corrected = np.minimum.accumulate(bh_threshold[::-1])[::-1]
    qvals = np.empty(n)
    qvals[ranked] = p_corrected
    return qvals


def archetype_correlation_with_behaviour(behavior_df, archetype_probs, metadata_cols):
    cols = [c for c in behavior_df.columns if c not in metadata_cols]
    behavior_data = behavior_df[cols].select_dtypes(include=[np.number])
    behavior_z = behavior_data.apply(zscore, nan_policy='omit').fillna(0)
    n_arch = archetype_probs.shape[1]
    results = []
    for a in range(n_arch):
        for col in behavior_z.columns:
            r, p = pearsonr(archetype_probs[:, a], behavior_z[col])
            results.append({'Archetype': f'Archetype{a+1}', 'Behavior': col, 'Pearson_r': r, 'p_value': p})
    df = pd.DataFrame(results)
    for a in range(n_arch):
        mask = df['Archetype'] == f'Archetype{a+1}'
        df.loc[mask, 'p_adjusted_BH'] = _bh_correction(df.loc[mask, 'p_value'].values)
    return df.sort_values(['Archetype', 'Pearson_r'], ascending=[True, False])


def hormone_correlation_with_archetypes(hormone_arch_df, metadata_cols):
    arch_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    hormone_cols = [c for c in hormone_arch_df.columns
                    if c not in metadata_cols and c not in arch_cols
                    and c not in ['PC1', 'PC2', 'Dominant_archetype']]
    results = []
    for a in arch_cols:
        for h in hormone_cols:
            mask = hormone_arch_df[a].notna() & hormone_arch_df[h].notna()
            if mask.sum() < 3:
                continue
            r, p = pearsonr(hormone_arch_df.loc[mask, a], hormone_arch_df.loc[mask, h])
            results.append({'Archetype': a.replace('_prob', ''), 'Hormone': h, 'Pearson_r': r, 'p_value': p})
    df = pd.DataFrame(results)
    for a in arch_cols:
        mask = df['Archetype'] == a.replace('_prob', '')
        df.loc[mask, 'p_adjusted_BH'] = _bh_correction(df.loc[mask, 'p_value'].values)
    return df.sort_values(['Archetype', 'Pearson_r'], ascending=[True, False])


def plot_pca_triangle(all_pca_coords, archetypes, counts, varexlp, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_pca_coords[:, 0], all_pca_coords[:, 1], alpha=0.5, s=25, c='steelblue', label='Every day')
    tri_x = [archetypes[0, 0], archetypes[1, 0], archetypes[2, 0], archetypes[0, 0]]
    tri_y = [archetypes[0, 1], archetypes[1, 1], archetypes[2, 1], archetypes[0, 1]]
    ax.plot(tri_x, tri_y, 'k-', linewidth=2, label='Archetype triangle')
    colors = ['red', 'green', 'orange']
    for v in range(3):
        ax.scatter(archetypes[v, 0], archetypes[v, 1], c=colors[v], s=150, marker='^', zorder=5)
        ax.annotate(f'Archetype {v+1}\n(n={counts[v]})', archetypes[v],
                    textcoords='offset points', xytext=(0, 15), ha='center', fontsize=10,
                    color=colors[v], fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA with Archetype Triangle (Var explained: {varexlp:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved triangle plot: {save_path}')


def plot_significant_correlations(corr_df, value_col, save_path, pval_col='p_adjusted_BH', threshold=0.05):
    sig = corr_df[corr_df[pval_col] < threshold].copy()
    max_n = max(len(sig[sig['Archetype'] == f'Archetype{a+1}']) for a in range(3))
    height = max(4, max_n * 0.25)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=False)
    for a_idx, ax in enumerate(axes):
        arch_label = f'Archetype{a_idx+1}'
        sub = sig[sig['Archetype'] == arch_label].sort_values(value_col, ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, 'No significant\ncorrelations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(arch_label)
            continue
        colors = ['green' if r > 0 else 'red' for r in sub[value_col].values]
        ax.barh(range(len(sub)), sub[value_col].values, color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.iloc[:, 1].values, fontsize=8)
        ax.axvline(0, color='grey', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Pearson r')
        ax.set_title(f'{arch_label} (n={len(sub)})')
    label = 'BH-adjusted' if pval_col == 'p_adjusted_BH' else 'raw'
    fig.suptitle(f'Significant Correlations ({label} p < {threshold})', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved significant correlations plot: {save_path}')


def main():
    data_dir = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use'
    behavior = 'Behavior_every_day.xlsx'
    hormones = 'Hormones.xlsx'

    behavior_df = pd.read_excel(f'{data_dir}/{behavior}')
    hormones_df = pd.read_excel(f'{data_dir}/{hormones}')

    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']

    print('Running PCA on behavior data...')
    pca_model, all_pca_coords, behavior_cols = run_pca(behavior_df)
    print(f'  PCA explained variance ratio: {pca_model.explained_variance_ratio_}')
    print(f'  Total variance explained by 2 PCs: {pca_model.explained_variance_ratio_.sum():.3f}')

    print('Finding best triangle with PCHA...')
    archetypes, varexlp, counts = find_best_triangle(all_pca_coords, n_trials=500)
    print(f'  Best triangle variance explained: {varexlp:.3f}')
    print(f'  Counts per vertex: {counts}')
    print(f'  Archetype coordinates (PC1, PC2):')
    for i, arch in enumerate(archetypes):
        print(f'    Archetype {i+1}: PC1={arch[0]:.4f}, PC2={arch[1]:.4f}')

    plot_pca_triangle(all_pca_coords, archetypes, counts, varexlp,
                      os.path.join(data_dir, 'pca_triangle.pdf'))

    print('Computing archetype probabilities...')
    prob_coeffs = compute_archetype_probabilities(all_pca_coords, archetypes)

    table_df = behavior_df[metadata_cols].copy()
    table_df['PC1'] = all_pca_coords[:, 0]
    table_df['PC2'] = all_pca_coords[:, 1]
    table_df['Archetype1_prob'] = prob_coeffs[:, 0]
    table_df['Archetype2_prob'] = prob_coeffs[:, 1]
    table_df['Archetype3_prob'] = prob_coeffs[:, 2]
    prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    table_df['Dominant_archetype'] = table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    print('Correlating archetypes with original behavior parameters...')
    corr_behav = archetype_correlation_with_behaviour(behavior_df, prob_coeffs, metadata_cols)
    print(corr_behav.groupby('Archetype').head(5).to_string(index=False))

    plot_significant_correlations(corr_behav, 'Pearson_r',
                                  os.path.join(data_dir, 'archetype_behaviour_correlations.pdf'))

    print('Joining with hormone data...')
    group_keys = ['Experiment', 'sex', 'Hierarchy']
    agg = table_df.groupby(group_keys).agg(
        PC1=('PC1', 'mean'),
        PC2=('PC2', 'mean'),
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    ).reset_index()
    cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
    agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    hormones_arch = agg.merge(
        hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
        on=['Experiment', 'sex', 'Hierarchy'], how='inner'
    )
    arch_prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    for a_idx, cname in enumerate(['cum_arch1', 'cum_arch2', 'cum_arch3']):
        if cname in hormones_arch.columns:
            hormones_arch[arch_prob_cols[a_idx]] = hormones_arch[cname]
            hormones_arch = hormones_arch.drop(columns=[cname])
    for c in ['n_days']:
        if c in hormones_arch.columns:
            hormones_arch = hormones_arch.drop(columns=[c])

    print('Correlating hormones with archetypes...')
    corr_horm = hormone_correlation_with_archetypes(hormones_arch, metadata_cols)
    print(corr_horm.groupby('Archetype').head(5).to_string(index=False))

    plot_significant_correlations(corr_horm, 'Pearson_r',
                                  os.path.join(data_dir, 'archetype_hormone_correlations_BH.pdf'))
    plot_significant_correlations(corr_horm, 'Pearson_r',
                                  os.path.join(data_dir, 'archetype_hormone_correlations_raw.pdf'),
                                  pval_col='p_value', threshold=0.05)

    output_dir = data_dir
    corr_behav.to_excel(os.path.join(output_dir, 'archetype_behaviour_correlation.xlsx'), index=False)
    corr_horm.to_excel(os.path.join(output_dir, 'archetype_hormone_correlation.xlsx'), index=False)
    table_df.to_csv(os.path.join(output_dir, 'archetype_probabilities_per_day.xlsx'), index=False)
    hormones_arch.to_csv(os.path.join(output_dir, 'hormones_with_archetypes.xlsx'), index=False)

    print(f'\nResults saved to {output_dir}')
    print(f'  - archetype_behaviour_correlation.xlsx')
    print(f'  - archetype_hormone_correlation.xlsx')
    print(f'  - archetype_probabilities_per_day.xlsx')
    print(f'  - hormones_with_archetypes.xlsx')
    print(f'  - pca_triangle.pdf')
    print(f'  - archetype_behaviour_correlations.pdf')
    print(f'  - archetype_hormone_correlations.pdf')

if __name__ == '__main__':
    main()
