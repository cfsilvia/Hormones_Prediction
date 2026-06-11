import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def _solve_convex_coeff(A, b):
    n = A.shape[1]
    x0 = np.ones(n) / n
    def loss(x):
        return np.sum((A @ x - b) ** 2)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1)] * n
    res = minimize(loss, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
    return res.x

def archetypal_analysis(X, k, seed=42, max_iter=200, tol=1e-6):
    np.random.seed(seed)
    n, d = X.shape
    idx = np.random.choice(n, k, replace=False)
    Z = X[idx].copy()
    for iteration in range(max_iter):
        Z_old = Z.copy()
        S = np.zeros((n, k))
        for i in range(n):
            S[i] = _solve_convex_coeff(Z.T, X[i])
        C = np.zeros((k, n))
        for j in range(k):
            C[j] = _solve_convex_coeff(X.T, Z[j])
        Z = C @ X
        diff = np.linalg.norm(Z - Z_old) / max(1e-8, np.linalg.norm(Z_old))
        if diff < tol:
            break
    return Z, S

# ================================================================
# 1. Load + Standardize
# ================================================================
df = pd.read_csv('Data_behaviour_per_day.csv')
feature_cols = df.columns[7:]
X_raw = df[feature_cols].values
col_names = feature_cols.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
n, m = X_scaled.shape

# ================================================================
# 2. PCA: choose optimal number of components
# ================================================================
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', color='#2c3e50', linewidth=2, markersize=5)
ax.axhline(0.8, color='#e74c3c', linestyle='--', alpha=0.7, label='80% variance')
ax.axhline(0.9, color='#e67e22', linestyle='--', alpha=0.7, label='90% variance')
n80 = np.where(cumvar >= 0.8)[0][0] + 1
n90 = np.where(cumvar >= 0.9)[0][0] + 1
ax.axvline(n80, color='#e74c3c', linestyle=':', alpha=0.5)
ax.axvline(n90, color='#e67e22', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of PCA components', fontsize=13)
ax.set_ylabel('Cumulative variance explained', fontsize=13)
ax.set_title('PCA variance explained', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ev = pca_full.explained_variance_ratio_ * 100
ax.bar(range(1, 21), ev[:20], color='#3498db', alpha=0.8, edgecolor='white')
ax.plot(range(1, 21), ev[:20], 'o-', color='#e74c3c', linewidth=1, markersize=4)
ax.set_xlabel('PC', fontsize=13)
ax.set_ylabel('Variance explained (%)', fontsize=13)
ax.set_title('Individual variance per PC (first 20)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pca_variance.png")

print(f"Components for 80% variance: {n80}")
print(f"Components for 90% variance: {n90}")
print(f"Components for 95% variance: {np.where(cumvar >= 0.95)[0][0] + 1}")

# Choose number of PCs: enough to explain most variance but not too many
# Use n80 as a reasonable trade-off
n_pc = n80
print(f"\nUsing {n_pc} PCA components (~80% variance)")

pca = PCA(n_components=n_pc)
X_pca = pca.fit_transform(X_scaled)

# ================================================================
# 3. Archetypal analysis on PCA-reduced data
# ================================================================
k = 4
Z_pca, S = archetypal_analysis(X_pca, k, seed=42)
sse = np.sum((X_pca - S @ Z_pca) ** 2)
print(f"k=4 SSE in PCA space: {sse:.4f}")

# Map archetypes back to original feature space for interpretation
# Z_original = Z_pca @ pca.components_ * std + mean
Z_original = pca.inverse_transform(Z_pca)

arch_colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
arch_labels = ['Arch 1', 'Arch 2', 'Arch 3', 'Arch 4']
arch_names_viz = ['Solitary\nHigh-Accel', 'Hiding/\nInactive', 'Social\nExplorer', 'Social\nContact']

print("\n=== Archetypes (PCA+AA, mapped to original features) ===")
for i in range(k):
    print(f"\n--- {arch_labels[i]} ---")
    z_scores = (Z_original[i] - scaler.mean_) / np.sqrt(scaler.var_)
    top_pos = np.argsort(z_scores)[-5:][::-1]
    top_neg = np.argsort(z_scores)[:5]
    print("  Highest:")
    for idx in top_pos:
        print(f"    {col_names[idx]}: {Z_original[i][idx]:.4f} (z={z_scores[idx]:.2f})")
    print("  Lowest:")
    for idx in top_neg:
        print(f"    {col_names[idx]}: {Z_original[i][idx]:.4f} (z={z_scores[idx]:.2f})")

# Coefficients table
df_coeff = pd.DataFrame(S, columns=arch_labels).round(4)
result = df[['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips','Animal']].copy()
result = pd.concat([result, df_coeff], axis=1)
result['Dominant'] = df_coeff.idxmax(axis=1)

print("\n=== By Hierarchy ===")
print(pd.crosstab(result['Hierarchy'], result['Dominant']).to_string())
print("\n=== By Sex ===")
print(pd.crosstab(result['sex'], result['Dominant']).to_string())

# ================================================================
# 4. BOOTSTRAP PROBABILITIES (on PCA data)
# ================================================================
print("\n=== Bootstrap probabilities (200 runs, PCA space) ===")
n_boot = 200
boot_S = np.zeros((n_boot, n, k))
for b in range(n_boot):
    boot_idx = np.random.choice(n, n, replace=True)
    X_boot = X_pca[boot_idx]
    try:
        Zb, Sb = archetypal_analysis(X_boot, k, seed=b)
        S_proj = np.zeros((n, k))
        for i in range(n):
            S_proj[i] = _solve_convex_coeff(Zb.T, X_pca[i])
        boot_S[b] = S_proj
    except:
        boot_S[b] = np.nan
boot_S = boot_S[~np.isnan(boot_S).any(axis=(1,2))]
print(f"Successful runs: {len(boot_S)}")

S_mean = boot_S.mean(axis=0)
S_std = boot_S.std(axis=0)
dominant_ref = np.argmax(S, axis=1)
dominant_boot = np.argmax(boot_S, axis=2)
dominant_prob = np.zeros((n, k))
for s in range(n):
    for a in range(k):
        dominant_prob[s, a] = (dominant_boot[:, s] == a).mean()

print(f"\n{'Sample':<30} {'Sex':<6} {'Hier':<8} {'Dom':<8} {'  S(dom) ± boot_std':<25} {'P(dom)':<8}")
print('-'*85)
for i in range(n):
    d = dominant_ref[i]
    print(f'{df["Experiment"].iloc[i][:7]:<7} {df["sex"].iloc[i][:1].upper():<6} '
          f'{df["Hierarchy"].iloc[i]:<8} {arch_labels[d]:<8}  '
          f'{S[i,d]:.3f} ± {S_std[i,d]:.3f}             {dominant_prob[i,d]:.3f}')

print(f"\nClassification confidence:")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
    c = (dominant_prob.max(axis=1) >= thresh).sum()
    print(f"  P(dom) >= {thresh:.1f}: {c}/{n} ({c/n*100:.0f}%)")

# ================================================================
# 5. VISUALIZATIONS
# ================================================================

# 5a. Pyramid by sex
fig, ax = plt.subplots(figsize=(10, 6))
mc = [((df['sex']=='male')&(result['Dominant']==a)).sum() for a in arch_labels]
fc = [((df['sex']=='female')&(result['Dominant']==a)).sum() for a in arch_labels]
mx = max(max(mc), max(fc)) + 1
for i in range(k):
    ax.barh(i, -mc[i], 0.6, color=arch_colors[i], alpha=0.85)
    ax.barh(i, fc[i], 0.6, color=arch_colors[i], alpha=0.85)
    ax.text(-mc[i]-0.15, i, str(mc[i]), ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(fc[i]+0.15, i, str(fc[i]), ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(0, i, arch_names_viz[i], ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=arch_colors[i], alpha=0.9))
ax.set_yticks(range(k)); ax.set_yticklabels(['']*k)
ax.set_xlim(-mx-1, mx+1); ax.set_xticks(range(-mx, mx+1))
ax.set_xticklabels([str(abs(x)) for x in range(-mx, mx+1)])
ax.axvline(0, color='black', linewidth=1.5)
ax.text(-mx*0.5-0.5, -0.5, 'Male', ha='center', fontsize=13, fontweight='bold')
ax.text(mx*0.5+0.5, -0.5, 'Female', ha='center', fontsize=13, fontweight='bold')
ax.set_title(f'Archetypes by Sex (PCA={n_pc}comps, AA=k=4)', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig('pca_pyramid.png', dpi=150, bbox_inches='tight'); plt.close()
print("\nSaved: pca_pyramid.png")

# 5b. 3D (project to 3 PCs for display)
pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(X_pca)
Z3 = pca3.transform(Z_pca)
var3 = pca3.explained_variance_ratio_ * 100
dom = result['Dominant'].values

for elev, azim, suffix in [(25, -45, ''), (15, 135, '_alt')]:
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(k):
        m = dom == arch_labels[i]
        for s, mk in [('male','o'),('female','^')]:
            mx = m & (df['sex']==s)
            if mx.sum(): 
                ax.scatter(X3[mx,0], X3[mx,1], X3[mx,2], c=[arch_colors[i]], marker=mk,
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    for i in range(k):
        ax.scatter(Z3[i,0], Z3[i,1], Z3[i,2], c=[arch_colors[i]], marker='D',
                   s=500, edgecolors='black', linewidth=2, zorder=10)
        ax.text(Z3[i,0], Z3[i,1], Z3[i,2], f'  {arch_labels[i]}', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var3[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var3[1]:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({var3[2]:.1f}%)', fontsize=12)
    ax.set_title(f'PCA+AA: Archetypes in 3D (PCA={n_pc} comps, AA=k=4)', fontsize=14, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(f'pca_3d{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: pca_3d{suffix}.png')

# 5c. Profiles
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i in range(k):
    zs = (Z_original[i] - scaler.mean_) / np.sqrt(scaler.var_)
    ax = axes[i]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in zs]
    ax.barh(range(len(col_names)), zs, color=colors)
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names, fontsize=7)
    ax.set_xlabel('Z-score')
    ax.set_title(f'{arch_labels[i]}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5); ax.invert_yaxis()
plt.tight_layout(); plt.savefig('pca_profiles.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: pca_profiles.png")

# 5d. Bootstrap probability heatmap
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(dominant_prob.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_yticks(range(k))
ax.set_yticklabels([f'Arch {i+1}' for i in range(k)], fontsize=11)
ax.set_xticks(range(n))
ax.set_xticklabels([f'{df["Experiment"].iloc[i][:5]}-{df["Hierarchy"].iloc[i][:3]}'
                     for i in range(n)], rotation=90, fontsize=5)
ax.set_title(f'Bootstrap P(assignment) | PCA={n_pc} comps, k=4', fontsize=14, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, shrink=0.7); cbar.set_label('Probability', fontsize=11)
plt.tight_layout(); plt.savefig('pca_probabilities.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: pca_probabilities.png")

# 5e. Bootstrap CIs for dominant coefficient
fig, ax = plt.subplots(figsize=(12, 4))
for i in range(n):
    d = dominant_ref[i]
    ax.errorbar(i, S_mean[i,d], yerr=S_std[i,d]*1.96, fmt='o',
                color=arch_colors[d], alpha=0.6, capsize=2, markersize=5)
ax.set_xticks(range(n))
ax.set_xticklabels([f'{df["Experiment"].iloc[i][:5]}-{df["Hierarchy"].iloc[i][:3]}'
                     for i in range(n)], rotation=90, fontsize=5)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Coefficient (95% CI)', fontsize=12)
ax.set_title('Dominant coefficient with bootstrap 95% CI', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('pca_bootstrap_ci.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: pca_bootstrap_ci.png")

# 5f. Stability comparison: original 54D vs PCA-reduced
print("\n=== Stability comparison: raw 54D vs PCA-reduced ===")
from scipy.spatial.distance import cdist

# Run AA on raw 54D
Z_raw, S_raw = archetypal_analysis(X_scaled, k, seed=42)
boot_raw = np.zeros((min(50, n_boot), n, k))
for b in range(min(50, n_boot)):
    bi = np.random.choice(n, n, replace=True)
    try:
        Zb, _ = archetypal_analysis(X_scaled[bi], k, seed=b)
        for i in range(n):
            boot_raw[b,i] = _solve_convex_coeff(Zb.T, X_scaled[i])
    except:
        boot_raw[b] = np.nan
boot_raw = boot_raw[~np.isnan(boot_raw).any(axis=(1,2))]
raw_std = boot_raw.std(axis=0).mean()
pca_std = S_std.mean()
print(f"  Mean std of coefficients: raw={raw_std:.3f} vs PCA={pca_std:.3f}")
print(f"  Improvement: {(1 - pca_std/raw_std)*100:.0f}% reduction in uncertainty")
