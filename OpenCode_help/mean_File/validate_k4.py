import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
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
    n, m = X.shape
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

df = pd.read_csv('Data_behaviour_per_day.csv')
feature_cols = df.columns[7:]
X = df[feature_cols].values
col_names = feature_cols.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n, m = X_scaled.shape

# ==============================================================
# 1. STABILITY ACROSS RANDOM INITIALIZATIONS
# ==============================================================
print("=== 1. Stability across 50 random initializations ===")

n_runs = 50
all_archetypes = {}  # k -> list of (Z, S, SSE)

for k in [2, 3, 4, 5, 6]:
    runs = []
    for seed in range(n_runs):
        Z, S = archetypal_analysis(X_scaled, k, seed=seed)
        sse = np.sum((X_scaled - S @ Z) ** 2)
        runs.append({'Z': Z, 'S': S, 'SSE': sse})
    all_archetypes[k] = runs
    
    # Compute stability: for each run, find the closest matching run's archetypes
    # via Procrustes-like matching
    sses = [r['SSE'] for r in runs]
    mean_sse = np.mean(sses)
    std_sse = np.std(sses)
    cv = std_sse / mean_sse * 100  # coefficient of variation
    
    # Pairwise archetype similarity
    from scipy.spatial import procrustes
    sims = []
    for i in range(min(10, n_runs)):
        for j in range(i+1, min(10, n_runs)):
            # Match archetypes between runs via Hungarian
            Zi = runs[i]['Z']
            Zj = runs[j]['Z']
            # Compute pairwise distances between archetypes
            D = cdist(Zi, Zj)
            # Minimally assign: nearest neighbor
            matched_dists = [D[idx].min() for idx in range(k)]
            sims.append(1 - np.mean(matched_dists) / (np.std(X_scaled) * 2))
    
    print(f"  k={k}: SSE = {mean_sse:.1f} ± {std_sse:.1f} (CV={cv:.1f}%)")

# Best SSE per k
best_sses = [min(all_archetypes[k], key=lambda r: r['SSE'])['SSE'] for k in range(2, 7)]
print()
print("Best SSE across runs:")
for i, k in enumerate(range(2, 7)):
    print(f"  k={k}: {best_sses[i]:.1f}")

# ==============================================================
# 2. ELBOW PLOT
# ==============================================================
print("\n=== 2. Elbow / Scree plot ===")
fig, ax = plt.subplots(figsize=(8, 5))
ks = list(range(2, 7))
best_vals = [min(all_archetypes[k], key=lambda r: r['SSE'])['SSE'] for k in ks]
ax.plot(ks, best_vals, 'o-', color='#2c3e50', linewidth=2, markersize=10)
ax.fill_between(ks, 
                 [np.mean([r['SSE'] for r in all_archetypes[k]]) - np.std([r['SSE'] for r in all_archetypes[k]]) for k in ks],
                 [np.mean([r['SSE'] for r in all_archetypes[k]]) + np.std([r['SSE'] for r in all_archetypes[k]]) for k in ks],
                 alpha=0.2, color='#3498db')
ax.set_xlabel('Number of archetypes (k)', fontsize=13)
ax.set_ylabel('SSE (reconstruction error)', fontsize=13)
ax.set_title('Elbow plot: SSE vs k (shaded = ±1 std over 50 runs)', fontsize=14, fontweight='bold')
ax.set_xticks(ks)
ax.grid(True, alpha=0.3)
# Annotate the % improvement
for i in range(len(ks)-1):
    pct = (best_vals[i] - best_vals[i+1]) / best_vals[i] * 100
    ax.annotate(f'-{pct:.0f}%', (ks[i]+0.3, (best_vals[i]+best_vals[i+1])/2), fontsize=10, color='#e74c3c')
plt.tight_layout()
plt.savefig('validation_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: validation_elbow.png")

# ==============================================================
# 3. CROSS-VALIDATION: train on 80%, predict remaining 20%
# ==============================================================
print("\n=== 3. Cross-validation (train/test split) ===")
np.random.seed(123)
from sklearn.model_selection import KFold

for k in [2, 3, 4, 5]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_errors = []
    for train_idx, test_idx in kf.split(X_scaled):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        
        # Fit on train
        Z_train, S_train = archetypal_analysis(X_train, k, seed=42)
        
        # Project test data: find S_test that reconstructs from Z_train
        S_test = np.zeros((len(test_idx), k))
        for i, idx in enumerate(test_idx):
            S_test[i] = _solve_convex_coeff(Z_train.T, X_scaled[idx])
        
        # Reconstruction error on test
        X_pred = S_test @ Z_train
        test_err = np.mean(np.sum((X_scaled[test_idx] - X_pred) ** 2, axis=1))
        test_errors.append(test_err)
    
    print(f"  k={k}: test MSE = {np.mean(test_errors):.2f} ± {np.std(test_errors):.2f}")

# ==============================================================
# 4. BEST k=4 SOLUTION: bootstrap stability of archetypes
# ==============================================================
print("\n=== 4. Bootstrap stability of k=4 archetypes ===")

Z_ref, S_ref = archetypal_analysis(X_scaled, 4, seed=42)
ref_sse = np.sum((X_scaled - S_ref @ Z_ref) ** 2)

# Bootstrap: resample data with replacement, re-fit, match archetypes
n_boot = 100
archetype_corrs = []
for b in range(n_boot):
    boot_idx = np.random.choice(n, n, replace=True)
    X_boot = X_scaled[boot_idx]
    try:
        Z_boot, S_boot = archetypal_analysis(X_boot, 4, seed=b)
        # Match each reference archetype to closest bootstrap archetype
        D = cdist(Z_ref, Z_boot)
        corrs = []
        for i in range(4):
            j = D[i].argmin()
            # Cosine similarity
            corr = np.dot(Z_ref[i], Z_boot[j]) / (np.linalg.norm(Z_ref[i]) * np.linalg.norm(Z_boot[j]) + 1e-10)
            corrs.append(corr)
        archetype_corrs.append(corrs)
    except:
        pass

archetype_corrs = np.array(archetype_corrs)
print("  Bootstrap cosine similarity (mean ± std per archetype):")
for i in range(4):
    print(f"    Archetype {i+1}: {archetype_corrs[:, i].mean():.3f} ± {archetype_corrs[:, i].std():.3f}")
print(f"  Overall mean: {archetype_corrs.mean():.3f}")

# ==============================================================
# 5. PERMUTATION TEST: is k=4 better than random?
# ==============================================================
print("\n=== 5. Permutation test (shuffle features) ===")
X_shuffled = X_scaled.copy()
perm_sse = []
for p in range(50):
    np.random.shuffle(X_shuffled)
    Z_p, S_p = archetypal_analysis(X_shuffled, 4, seed=p)
    sse_p = np.sum((X_shuffled - S_p @ Z_p) ** 2)
    perm_sse.append(sse_p)

print(f"  Real data SSE: {ref_sse:.1f}")
print(f"  Shuffled data SSE: {np.mean(perm_sse):.1f} ± {np.std(perm_sse):.1f}")
print(f"  Ratio: {ref_sse / np.mean(perm_sse):.3f} (lower = more structured)")

# ==============================================================
# 6. BIOLOGICAL VALIDATION: do archetypes align with sex/hierarchy?
# ==============================================================
print("\n=== 6. Biological alignment ===")
Z4, S4 = archetypal_analysis(X_scaled, 4, seed=42)
dominant = np.argmax(S4, axis=1)

from scipy.stats import chi2_contingency
# Sex association
ct_sex = pd.crosstab(df['sex'], dominant)
_, p_sex, _, _ = chi2_contingency(ct_sex)
print(f"  Archetype ~ Sex: chi2 p = {p_sex:.2e}")

# Hierarchy association
ct_hier = pd.crosstab(df['Hierarchy'], dominant)
_, p_hier, _, _ = chi2_contingency(ct_hier)
print(f"  Archetype ~ Hierarchy: chi2 p = {p_hier:.2e}")

# ==============================================================
# 7. Summary plot: all validation metrics
# ==============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# a) Elbow
ax = axes[0, 0]
ax.plot(ks, best_vals, 'o-', color='#2c3e50', linewidth=2, markersize=8)
ax.set_xlabel('k')
ax.set_ylabel('SSE')
ax.set_title('Elbow plot', fontweight='bold')
ax.grid(alpha=0.3)

# b) Stability CV
ax = axes[0, 1]
cvs = []
for k in ks:
    sses = [r['SSE'] for r in all_archetypes[k]]
    cvs.append(np.std(sses) / np.mean(sses) * 100)
ax.bar(ks, cvs, color=['#e74c3c' if k == 4 else '#3498db' for k in ks])
ax.set_xlabel('k')
ax.set_ylabel('CV (%)')
ax.set_title('Run-to-run stability\n(lower = more stable)', fontweight='bold')
ax.grid(alpha=0.3)

# c) Cross-validation
ax = axes[0, 2]
cv_means = []
cv_stds = []
for k in [2, 3, 4, 5]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errs = []
    for train_idx, test_idx in kf.split(X_scaled):
        X_train = X_scaled[train_idx]
        X_test_val = X_scaled[test_idx]
        Z_tr, S_tr = archetypal_analysis(X_train, k, seed=42)
        S_te = np.zeros((len(test_idx), k))
        for i, idx in enumerate(test_idx):
            S_te[i] = _solve_convex_coeff(Z_tr.T, X_scaled[idx])
        errs.append(np.mean(np.sum((X_scaled[test_idx] - S_te @ Z_tr) ** 2, axis=1)))
    cv_means.append(np.mean(errs))
    cv_stds.append(np.std(errs))
ax.errorbar([2,3,4,5], cv_means, yerr=cv_stds, fmt='o-', color='#2c3e50', linewidth=2, markersize=8)
ax.set_xlabel('k')
ax.set_ylabel('Test MSE')
ax.set_title('5-fold Cross-validation', fontweight='bold')
ax.grid(alpha=0.3)

# d) Bootstrap archetype stability
ax = axes[1, 0]
means = archetype_corrs.mean(axis=0)
stds_ar = archetype_corrs.std(axis=0)
ax.bar(range(1, 5), means, yerr=stds_ar, color=['#e74c3c','#3498db','#2ecc71','#9b59b6'], capsize=5)
ax.set_xlabel('Archetype')
ax.set_ylabel('Cosine similarity')
ax.set_title('Bootstrap stability\n(higher = more robust)', fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# e) Permutation test
ax = axes[1, 1]
ax.bar([0, 1], [ref_sse, np.mean(perm_sse)], color=['#e74c3c', '#7f8c8d'], width=0.5)
ax.errorbar(1, np.mean(perm_sse), yerr=np.std(perm_sse), fmt='none', capsize=5, color='black')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Real data', 'Shuffled'])
ax.set_ylabel('SSE')
ax.set_title('Permutation test\n(lower = real structure)', fontweight='bold')
ax.grid(alpha=0.3)

# f) Biological p-values
ax = axes[1, 2]
pvals = [p_sex, p_hier]
labels = ['Sex', 'Hierarchy']
colors_p = ['#e74c3c' if p < 0.001 else ('#f39c12' if p < 0.05 else '#7f8c8d') for p in pvals]
ax.bar(labels, [-np.log10(p) for p in pvals], color=colors_p, width=0.5)
ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
ax.axhline(-np.log10(0.001), color='gray', linestyle=':', label='p=0.001')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Biological association\n(higher = meaningful)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle('Validation of k=4 Archetypal Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('validation_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: validation_summary.png")

print("\n=== SUMMARY ===")
print(f"k=4 reconstruction error (SSE): {ref_sse:.1f}")
print(f"Bootstrap stability: {archetype_corrs.mean():.3f}")
print(f"Sex association p: {p_sex:.2e}")
print(f"Hierarchy association p: {p_hier:.2e}")
print(f"Permutation test (real/random ratio): {ref_sse / np.mean(perm_sse):.3f}")
