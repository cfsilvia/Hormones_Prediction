import pandas as pd
import numpy as np
from py_pcha import PCHA
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
import os
from archetype_analysis import compute_pca, compute_archetype_probabilities, select_top_per_archetype
from statsmodels.stats.multitest import multipletests



# input: df (DataFrame)
# output: df (DataFrame) — column names with spaces replaced by dots
def _normalize_column_names(df):
    df = df.copy()
    df.columns = [c.replace(' ', '.') for c in df.columns]
    return df

# input: pca_coords (ndarray), archetypes (ndarray)
# output: counts (list of int) — number of points assigned to each archetype vertex
def assign_to_nearest_vertex(pca_coords, archetypes):
    dists = np.linalg.norm(pca_coords[:, np.newaxis, :] - archetypes[np.newaxis, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    counts = [int(np.sum(assignments == v)) for v in range(archetypes.shape[0])]
    return counts

# input: data_dir (str) — path to folder with Excel files
# output: behavior_df (DataFrame), hormones_df (DataFrame)
def load_data(data_dir):
    behavior_df = pd.read_excel(f'{data_dir}/Behavior_every_day.xlsx')
    hormones_df = pd.read_excel(f'{data_dir}/Hormones.xlsx')
    return behavior_df, hormones_df

# input: data_dir (str) — path to folder with hormones_with_archetypes.xlsx
# output: hormones_arch (DataFrame)
def load_hormones_archetypes(data_dir):
    file_path = os.path.join(data_dir, 'hormones_with_archetypes.xlsx')
    hormones_arch = pd.read_excel(file_path)
    print(f'  Loaded hormones with archetypes: {file_path}')
    return hormones_arch

# input: behavior_df (DataFrame), data_dir (str)
# output: (archetypes, all_pca_coords, prob_coeffs, table_df, metadata_cols)
def run_archetype_analysis(behavior_df, data_dir):
    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']

    print('Running PCA...')
    pca_model, all_pca_coords, behavior_cols = run_pca(behavior_df)
    print(f'  Variance explained: {pca_model.explained_variance_ratio_.sum():.3f}')

    print('Finding best triangle with PCHA...')
    archetypes, varexlp, counts = find_best_triangle(all_pca_coords, n_trials=500)
    print(f'  Var explained: {varexlp:.3f}, Counts: {counts}')
    for i, arch in enumerate(archetypes):
        print(f'  Archetype {i+1}: PC1={arch[0]:.4f}, PC2={arch[1]:.4f}')

    plot_pca_triangle(all_pca_coords, archetypes, counts, varexlp,
                      os.path.join(data_dir, 'pca_triangle.pdf'),
                      sex_labels=behavior_df['sex'].values,
                      var_ratio=pca_model.explained_variance_ratio_)

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

    return archetypes, all_pca_coords, prob_coeffs, table_df, metadata_cols

# input: data_df (DataFrame) — behavior data with metadata columns
# output: (pca_model, pca_coords, behavior_cols)
def run_pca(data_df):
    return compute_pca(data_df)

# input: pca_coords (ndarray), n_trials (int)
# output: (best_archetypes, best_varexlp, best_counts) — best PCHA triangle
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

# input: all_pca_coords (ndarray), archetypes (ndarray), counts (list), varexlp (float), save_path (str), sex_labels (array-like, optional)
# output: None (saves PCA scatter + archetype triangle plot)
def plot_pca_triangle(all_pca_coords, archetypes, counts, varexlp, save_path, sex_labels=None, var_ratio=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if sex_labels is not None:
        sex_arr = np.array(sex_labels)
        female_mask = sex_arr == 'female'
        male_mask = ~female_mask
        ax.scatter(all_pca_coords[male_mask, 0], all_pca_coords[male_mask, 1],
                   alpha=0.5, s=25, c='steelblue', marker='o', label=f'Male ({male_mask.sum()})')
        ax.scatter(all_pca_coords[female_mask, 0], all_pca_coords[female_mask, 1],
                   alpha=0.5, s=25, c='red', marker='^', label=f'Female ({female_mask.sum()})')
    else:
        ax.scatter(all_pca_coords[:, 0], all_pca_coords[:, 1], alpha=0.5, s=25, c='steelblue', label='Every day')
    tri_x = [archetypes[0, 0], archetypes[1, 0], archetypes[2, 0], archetypes[0, 0]]
    tri_y = [archetypes[0, 1], archetypes[1, 1], archetypes[2, 1], archetypes[0, 1]]
    ax.plot(tri_x, tri_y, 'k-', linewidth=2, label='Archetype triangle')
    colors = ['red', 'green', 'orange']
    y_range = archetypes[:, 1].max() - archetypes[:, 1].min()
    y_mid = archetypes[:, 1].mean()
    for v in range(3):
        ax.scatter(archetypes[v, 0], archetypes[v, 1], c=colors[v], s=150, marker='^', zorder=5)
        offset_y = -15 if archetypes[v, 1] > y_mid + 0.1 * y_range else 10
        ax.annotate(f'Archetype {v+1}\n(n={counts[v]})', archetypes[v],
                    textcoords='offset points', xytext=(0, offset_y), ha='center', fontsize=10,
                    color=colors[v], fontweight='bold')
    pc1_label = f'PC1 ({var_ratio[0]*100:.1f}%)' if var_ratio is not None else 'PC1'
    pc2_label = f'PC2 ({var_ratio[1]*100:.1f}%)' if var_ratio is not None else 'PC2'
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_title(f'PCA with Archetype Triangle (Var explained: {varexlp:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved triangle plot: {save_path}')

# input: behavior_df (DataFrame), prob_coeffs (ndarray), metadata_cols (list), data_dir (str)
# output: corr_df (DataFrame) — behaviour correlations + plot
def correlate_with_behaviour(behavior_df, prob_coeffs, metadata_cols, data_dir):
    print('Correlating archetypes with behaviour...')
    corr_df = archetype_correlation_with_behaviour(behavior_df, prob_coeffs, metadata_cols)
    print(corr_df.groupby('Archetype').head(5).to_string(index=False))
    plot_significant_correlations(corr_df, 'Pearson_r',
                                  os.path.join(data_dir, 'archetype_behaviour_correlations.pdf'))
    return corr_df


# input: behavior_df (DataFrame), archetype_probs (ndarray), metadata_cols (list)
# output: df (DataFrame) — Pearson r, p_value, p_adjusted_BH per behaviour per archetype
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

# input: corr_df (DataFrame), value_col (str), save_path (str), pval_col (str), threshold (float)
# output: None (saves bar plot of significant correlations per archetype)
def plot_significant_correlations_BH(corr_df, value_col, save_path, pval_col='p_adjusted_BH', threshold=0.05):
    sig = corr_df[corr_df[pval_col] < threshold].copy()
    max_n = max(len(sig[sig['Archetype'] == f'Archetype{a+1}']) for a in range(3))
    height = max(4, max_n * 0.25)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=False)
    arch_colors = ['red', 'green', 'orange']
    for a_idx, ax in enumerate(axes):
        arch_label = f'Archetype{a_idx+1}'
        sub = sig[sig['Archetype'] == arch_label].sort_values(value_col, ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, 'No significant\ncorrelations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(arch_label)
            continue
        ax.barh(range(len(sub)), sub[value_col].values, color=arch_colors[a_idx], alpha=0.5)
        ax.set_yticks(range(len(sub)))
        label_col = 'Hormone' if 'Hormone' in sub.columns else 'Behavior' if 'Behavior' in sub.columns else sub.columns[1]
        ax.set_yticklabels(sub[label_col].values, fontsize=8)
        ax.axvline(0, color='grey', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Pearson r')
        ax.set_title(f'{arch_label} (n={len(sub)})')
    label = 'BH-adjusted' if pval_col == 'p_adjusted_BH' else 'raw'
    fig.suptitle(f'Significant Correlations ({label} p < {threshold})', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved significant correlations plot: {save_path}')

# input: pvals (array-like) — raw p-values
# output: qvals (ndarray) — BH-adjusted p-values
def _bh_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan)
    valid = np.isfinite(pvals)
    if np.any(valid):
        qvals[valid] = multipletests(pvals[valid], method='fdr_bh')[1]
    return qvals

# input: corr_df (DataFrame), value_col (str), save_path (str), pval_col (str), threshold (float)
# output: None (saves bar plot of significant correlations per archetype)
def plot_significant_correlations(corr_df, value_col, save_path, pval_col='p_value', threshold=0.05):
    sig = corr_df[corr_df[pval_col] < threshold].copy()
    max_n = max(len(sig[sig['Archetype'] == f'Archetype{a+1}']) for a in range(3))
    height = max(4, max_n * 0.25)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=False)
    arch_colors = ['red', 'green', 'orange']
    for a_idx, ax in enumerate(axes):
        arch_label = f'Archetype{a_idx+1}'
        sub = sig[sig['Archetype'] == arch_label].sort_values(value_col, ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, 'No significant\ncorrelations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(arch_label)
            continue
        ax.barh(range(len(sub)), sub[value_col].values, color=arch_colors[a_idx], alpha=0.5)
        ax.set_yticks(range(len(sub)))
        label_col = 'Hormone' if 'Hormone' in sub.columns else 'Behavior' if 'Behavior' in sub.columns else sub.columns[1]
        ax.set_yticklabels(sub[label_col].values, fontsize=8)
        ax.axvline(0, color='grey', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Pearson r')
        ax.set_title(f'{arch_label} (n={len(sub)})')
    fig.suptitle(f'Significant Correlations (p < {threshold})', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved significant correlations plot: {save_path}')


# input: table_df (DataFrame), hormones_df (DataFrame), metadata_cols (list), archetypes (ndarray), data_dir (str)
# output: hormones_arch (DataFrame) — per-animal archetype probs + hormone levels
def aggregate_and_merge_hormones(table_df, hormones_df, metadata_cols, archetypes, data_dir=None):
    print('Aggregating per animal and merging with hormones...')
    top_df = select_top_per_archetype(table_df, archetypes)

    hormones_arch = top_df.merge(
        hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
        on=['Experiment', 'sex', 'Hierarchy'], how='inner'
    )

    drop_cols = ['n_days', 'Type', 'Animal', 'Genotype', 'Mice.chips', 'Mice chips']
    drop_cols = [c for c in hormones_arch.columns if c in drop_cols or c.startswith('cum_')]
    hormones_arch = hormones_arch.drop(columns=drop_cols)

    if data_dir is not None:
        save_path = os.path.join(data_dir, 'hormones_with_archetypes.xlsx')
        hormones_arch.to_excel(save_path, index=False)
        print(f'  Saved hormones with archetypes: {save_path}')

    return hormones_arch

# input: hormones_arch (DataFrame), metadata_cols (list), data_dir (str)
# output: corr_df (DataFrame) — hormone correlations by sex + raw p-value plots
def correlate_with_hormones(hormones_arch, metadata_cols, data_dir):
    print('Correlating hormones with archetypes...')
    corr_df = hormone_correlation_with_archetypes_by_sex(hormones_arch, metadata_cols)
    print(corr_df.groupby(['sex', 'Archetype']).head(5).to_string(index=False))

    save_path = os.path.join(data_dir, 'archetype_hormone_correlation_by_sex.xlsx')
    corr_df.to_excel(save_path, index=False)
    print(f'  Saved hormone correlations by sex: {save_path}')

    for sex_value, sex_corr in corr_df.groupby('sex'):
        safe_sex = str(sex_value).replace(' ', '_')
        plot_significant_correlations(sex_corr, 'Pearson_r',
                                      os.path.join(data_dir, f'archetype_hormone_correlations_{safe_sex}_raw.pdf'),
                                      pval_col='p_value', threshold=0.1)
    return corr_df

# input: hormone_arch_df (DataFrame), metadata_cols (list)
# output: df (DataFrame) — per-sex Pearson r, p_value, p_adjusted_BH per hormone per archetype
def hormone_correlation_with_archetypes_by_sex(hormone_arch_df, metadata_cols):
    arch_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    missing_arch = [c for c in arch_cols if c not in hormone_arch_df.columns]
    if missing_arch:
        raise ValueError(f'Missing archetype probability columns: {missing_arch}')

    sex_col = 'sex' if 'sex' in hormone_arch_df.columns else 'Sex'
    if sex_col not in hormone_arch_df.columns:
        raise ValueError('Missing sex column')

    last_arch_idx = max(hormone_arch_df.columns.get_loc(c) for c in arch_cols)
    hormone_cols = [c for c in hormone_arch_df.columns[last_arch_idx + 1:]
                    if c != 'Dominant_archetype' and pd.api.types.is_numeric_dtype(hormone_arch_df[c])]

    results = []
    for sex_value, sex_df in hormone_arch_df.groupby(sex_col):
        for a in arch_cols:
            for h in hormone_cols:
                mask = sex_df[a].notna() & sex_df[h].notna()
                if mask.sum() < 3:
                    continue
                x = sex_df.loc[mask, a]
                y = sex_df.loc[mask, h]
                if x.nunique() < 2 or y.nunique() < 2:
                    continue
                r, p = pearsonr(x, y)
                results.append({'sex': sex_value, 'Archetype': a.replace('_prob', ''),
                                'Hormone': h, 'Pearson_r': r, 'p_value': p, 'n': int(mask.sum())})

    if not results:
        return pd.DataFrame(columns=['sex', 'Archetype', 'Hormone', 'Pearson_r', 'p_value', 'p_adjusted_BH', 'n'])

    df = pd.DataFrame(results)
    df['p_adjusted_BH'] = np.nan
    for _, group in df.groupby(['sex', 'Archetype']):
        df.loc[group.index, 'p_adjusted_BH'] = _bh_correction(group['p_value'].values)
    return df.sort_values(['sex', 'Archetype', 'Pearson_r'], ascending=[True, True, False])
