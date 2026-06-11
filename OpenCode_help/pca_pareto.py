import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f as f_dist, entropy, pearsonr
from scipy.spatial import ConvexHull, distance
from archetypal import archetypal_analysis, simplex_ls, project_onto_convex_polygon

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_excel(path):
    df = pd.read_excel(path)
    print(f"Loaded {path} - {df.shape}")
    return df


def align_columns(df_ref, df_other):
    def norm(c):
        return c.replace(' ', '.').replace('(', '.').replace(')', '').replace('..', '.').strip('.')
    rev = {norm(c): c for c in df_other.columns}
    out = np.zeros((len(df_other), len(df_ref.columns)))
    for i, c in enumerate(df_ref.columns):
        c2 = rev.get(norm(c))
        if c2:
            out[:, i] = df_other[c2].values.astype(float)
    return out


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return scaler, pca, X_scaled, X_pca


def pca_diagnostics(X_scaled, pca_2d, n_objects):
    pca_all = PCA().fit(X_scaled)
    ev_full = pca_all.explained_variance_ratio_
    cumvar = np.cumsum(ev_full)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"PCA fidelity:  90% var at {n90} PCs,  95% at {n95} PCs")
    #fidelity of reconstruction in 2D
    X_hat = pca_2d.inverse_transform(pca_2d.transform(X_scaled))
    spe = np.sum((X_scaled - X_hat) ** 2, axis=1)
    T2 = np.sum((pca_2d.transform(X_scaled) / np.sqrt(np.maximum(pca_2d.explained_variance_, 1e-15))) ** 2, axis=1)
    T2_limit = 2 * (n_objects - 1) / (n_objects - 2) * f_dist.ppf(0.95, 2, n_objects - 2)
    spe_limit = np.percentile(spe, 95)
    print(f"  Hotelling T^2 limit (95%): {T2_limit:.2f}  points above: {(T2 > T2_limit).sum()}")
    print(f"  SPE limit (95%): {spe_limit:.2f}  points above: {(spe > spe_limit).sum()}")
    print(f"  Kaiser criterion (eig > 1): {(pca_all.explained_variance_ > 1.0).sum()} PCs")

    return cumvar, n90, n95, spe, spe_limit


def top_loadings(pca_2d, param_names, n=5):
    loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
    for pc, label in [(0, 'PC1'), (1, 'PC2')]:
        idx = np.argsort(np.abs(loadings[:, pc]))[::-1][:n]
        print(f"  Top {label}: {[param_names[i] for i in idx]}")


# ---------------------------------------------------------------------------
# Archetype metrics
# ---------------------------------------------------------------------------

def archetype_metrics(X_pca, Z, A):
    hull_data = ConvexHull(X_pca)
    hull_arch = ConvexHull(Z)
    area_ratio = hull_arch.volume / hull_data.volume

    closest = distance.cdist(X_pca, Z).argmin(axis=0)
    ent = entropy(A.T + 1e-15, base=Z.shape[0])

    print(f"Archetype hull area / data hull area = {area_ratio:.1%}")
    print(f"Archetype closest data-point indices: {closest}")
    print(f"Mean mixing entropy: {ent.mean():.3f}  (0 = pure, 1 = fully mixed)")

    return hull_arch, area_ratio, closest, ent


# ---------------------------------------------------------------------------
# Robust Archetypal Analysis
# ---------------------------------------------------------------------------

def robust_archetypes(X, k, n_iter=20, sample_frac=0.8, tol=1e-4, verbose=True):
    """Run PCHA on random subsamples and return consensus archetypes.

    Each iteration samples `sample_frac` of points, runs PCHA,
    then all archetype coordinates are collected and clustered
    with k-means to resolve label switching.
    """
    from sklearn.cluster import KMeans
    n = X.shape[0]
    subsample_size = int(n * sample_frac)
    all_Z = []

    for it in range(n_iter):
        idx = np.random.default_rng(it).choice(n, subsample_size, replace=False)
        hull = ConvexHull(X[idx])
        Z_sub, _, _ = archetypal_analysis(X[idx], k, hull=hull, max_iter=20, tol=tol, verbose=False)
        all_Z.append(Z_sub)
        if verbose and (it + 1) % max(1, n_iter // 5) == 0:
            print(f"  Bootstrap {it+1}/{n_iter}")

    all_Z = np.vstack(all_Z)
    km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(all_Z)
    Z_robust = km.cluster_centers_

    # Ensure archetypes are inside convex hull
    hull_full = ConvexHull(X)
    for j in range(k):
        Z_robust[j] = project_onto_convex_polygon(Z_robust[j], hull_full)

    # Refine: fix Z, compute A for all points
    A_robust = np.zeros((n, k))
    for i in range(n):
        A_robust[i] = simplex_ls(Z_robust, X[i])

    err = np.sum((X - A_robust @ Z_robust) ** 2)
    return Z_robust, A_robust, err


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(X_pca, X2_pca, Z, A, hull_arch, evr, cumvar, n90, n95, spe, spe_limit, spe2, save_dir='.'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', alpha=0.4, s=25, label='Data (n=360)')
    ax.scatter(X2_pca[:, 0], X2_pca[:, 1], c='limegreen', alpha=0.6, s=30,
               marker='s', edgecolors='k', linewidths=0.3, label='Mean-of-the-days (n=60)')
    ax.scatter(Z[:, 0], Z[:, 1], c='crimson', s=100, marker='D',
               edgecolors='k', linewidths=0.8, zorder=5, label='Archetypes')
    for i, z in enumerate(Z):
        ax.annotate(f'A{i+1}', z, textcoords='offset points', xytext=(8, 8),
                    fontsize=11, fontweight='bold', color='crimson')
    for s in hull_arch.simplices:
        ax.plot(Z[s, 0], Z[s, 1], 'crimson', lw=2, alpha=0.8)
    ax.plot(Z[hull_arch.vertices, 0], Z[hull_arch.vertices, 1],
            'crimson', lw=1.5, alpha=0.5, label='Archetype hull')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'PC1 ({evr[0]:.1%})'); ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
    ax.set_title(f'PCHA — Archetypes (k={Z.shape[0]})')
    ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, 'o-', color='steelblue', markersize=3)
    ax.axhline(90, color='green', ls='--', alpha=0.5, label='90%')
    ax.axhline(95, color='crimson', ls='--', alpha=0.5, label='95%')
    ax.axvline(n90, color='green', ls=':', alpha=0.4)
    ax.axvline(n95, color='crimson', ls=':', alpha=0.4)
    ax.set_xlabel('Number of PCs'); ax.set_ylabel('Cumulative variance (%)')
    ax.set_title('Scree Plot'); ax.legend(fontsize=8); ax.set_xlim(0, 20); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=spe, cmap='YlOrRd', s=30,
                    alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.scatter(X2_pca[:, 0], X2_pca[:, 1], c=spe2, cmap='YlOrRd', s=30,
               marker='s', alpha=0.8, edgecolors='k', linewidths=0.3)
    ax.scatter(Z[:, 0], Z[:, 1], c='crimson', s=80, marker='D', edgecolors='k', linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label='SPE (Q residual)')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'Q Residuals (limit={spe_limit:.2f})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ss_res = np.sum((X_pca - A @ Z) ** 2, axis=1)
    ss_tot = np.sum((X_pca - X_pca.mean(axis=0)) ** 2, axis=1)
    r2 = 1 - ss_res / ss_tot
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=r2, cmap='viridis', s=30,
                    alpha=0.7, edgecolors='k', linewidths=0.3, vmin=0, vmax=1)
    ax.scatter(X2_pca[:, 0], X2_pca[:, 1], c='limegreen', s=30, marker='s',
               alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.scatter(Z[:, 0], Z[:, 1], c='crimson', s=80, marker='D', edgecolors='k', linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label='R^2 per point')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'PCHA reconstruction R^2')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pca_archetypes.pdf'), dpi=300); plt.close()
    print(f"Saved pca_archetypes.pdf in {save_dir}")


def plot_correlations(A, X, param_names, save_dir='.'):
    k = A.shape[1]
    all_corr = np.zeros((k, X.shape[1]))
    all_pvals = np.zeros((k, X.shape[1]))
    for j in range(k):
        for p in range(X.shape[1]):
            r, pv = pearsonr(A[:, j], X[:, p])
            all_corr[j, p] = r
            all_pvals[j, p] = pv

    labels = [f'A{i+1}' for i in range(k)]
    for j in range(k):
        idx = np.argsort(-np.abs(all_corr[j]))[:10]
        print(f"\n  {labels[j]} — top correlated parameters:")
        for rank, p in enumerate(idx, 1):
            print(f"    {rank:2d}.  {'+' if all_corr[j, p] > 0 else '-'}  r={all_corr[j, p]:+.3f}  {param_names[p]}")

    p_flat = all_pvals.ravel()
    m = len(p_flat)
    order = np.argsort(p_flat)
    p_sorted = p_flat[order]
    bh_thresh = np.arange(1, m + 1) / m * 0.05
    max_sig = np.where(p_sorted <= bh_thresh)[0]
    if len(max_sig) > 0:
        p_cutoff = p_sorted[max_sig[-1]]
    else:
        p_cutoff = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for j in range(k):
        ax = axes[j]
        mask = all_pvals[j] <= p_cutoff
        vars_ = np.where(mask)[0]
        vals = all_corr[j, vars_]
        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No significant correlations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Archetype {j+1}', fontsize=12, fontweight='bold')
            continue
        order = np.argsort(vals)
        vals = vals[order]
        names = [param_names[vars_[i]] for i in order]
        colors = ['crimson' if v > 0 else 'steelblue' for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.8, edgecolor='gray', linewidth=0.3)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color='gray', lw=0.8); ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel('Pearson r', fontsize=9)
        ax.set_title(f'Archetype {j+1}  ({len(vals)} sig.)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.2, axis='x')
    fig.suptitle('Archetype × Parameter Correlations (BH FDR less than 0.05)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pca_archetypes_corr.pdf'), dpi=300); plt.close()
    print(f"Saved pca_archetypes_corr.pdf in {save_dir}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ARCH_LABELS = {
    1: 'Social — high snout contacts, time together, approaching',
    2: 'High-Intensity — high running velocity, chasing, being chased',
    3: 'Sedentary — high sleep, low movement, walking & distance',
}

def print_summary(evr, err, area_ratio, ent_mean, Z, closest):
    print(f"\n=== Summary ===")
    print(f"PCA: 2 PCs explain {evr.sum():.1%} variance")
    print(f"Archetypes: {Z.shape[0]}, reconstruction error={err:.3f}")
    print(f"Archetype hull / data hull area = {area_ratio:.1%}")
    print(f"Mean mixing entropy: {ent_mean:.3f}")
    print(f"Archetypes (PC coordinates):")
    for i, z in enumerate(Z):
        print(f"  A{i+1}: ({z[0]:+.3f}, {z[1]:+.3f})  closest point = {closest[i]}  — {ARCH_LABELS[i+1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
   #=============
   #User settings
    #=============
    directory = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data'
    input_file = 'selected_columns_total_data_behaviour_without_repetitions.xlsx'
    input_file_mean_days = 'Data_behaviour_per_day_without_repetitions.xlsx'
    #=================
    total_data = load_excel(os.path.join(directory, input_file))
    df = total_data.copy()
    df = df.drop(columns=['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal'])  # Drop non-numeric columns
    metadata_cols = total_data[['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']]

    X = df.select_dtypes(include='number').values.astype(float)
    param_names = df.columns.tolist() 
    #============================================
    #load per-day data and align columns
    total_data_mean_days = load_excel(os.path.join(directory, input_file_mean_days))
    df2 = total_data_mean_days.copy()
    df2 = df2.drop(columns=['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']) 
    metadata_cols2 = total_data_mean_days[['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']]
    
    #align columns of df2 to match df
    X2 = align_columns(df, df2)
    print(f"  Aligned {X2.shape[1]} / {df.shape[1]} columns")
    ####################################################################################
    scaler, pca, X_scaled, X_pca = fit_pca(X)
    X2_scaled = scaler.transform(X2)
    X2_pca = pca.transform(X2_scaled)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}, total={evr.sum():.1%}")

    cumvar, n90, n95, spe, spe_limit = pca_diagnostics(X_scaled, pca, X.shape[0])
    top_loadings(pca, param_names)
#################################################
    k_arch = 3
    print(f"\n=== Archetypal Analysis (k={k_arch}) ===")
    Z, A, err = robust_archetypes(X_pca, k_arch, n_iter=20, verbose=True)
    print(f"Final reconstruction error (robust): {err:.4f}")

    # ---- compute mixing coefficients for per-day points ----
    A2 = np.zeros((X2_pca.shape[0], k_arch))
    for i in range(X2_pca.shape[0]):
        A2[i] = simplex_ls(Z, X2_pca[i])
    df_out = metadata_cols2.reset_index(drop=True).copy()
    for j in range(k_arch):
        df_out[f'Archetype_{j+1}'] = A2[:, j]
    arch_idx = A2.argmax(axis=1) + 1
    df_out['Archetype_number'] = 'Archetype_' + arch_idx.astype(str)
    df_out['Archetype_description'] = [ARCH_LABELS[i] for i in arch_idx]
    out_path = os.path.join(directory, 'archetype_coefficients_per_day.xlsx')
    df_out.to_excel(out_path, index=False)
    print(f"Saved {out_path}")
    #========================================================
    hull_arch, area_ratio, closest, ent = archetype_metrics(X_pca, Z, A)

    X2_hat = pca.inverse_transform(X2_pca)
    spe2 = np.sum((X2_scaled - X2_hat) ** 2, axis=1)

    plot_overview(X_pca, X2_pca, Z, A, hull_arch, evr, cumvar, n90, n95, spe, spe_limit, spe2, save_dir=directory)

    print(f"\n=== Archetype × Parameter Correlations ===")
    plot_correlations(A, X, param_names, save_dir=directory)

    print_summary(evr, err, area_ratio, ent.mean(), Z, closest)


if __name__ == '__main__':
    main()
