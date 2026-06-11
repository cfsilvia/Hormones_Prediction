import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# --- Load & prepare ---
df = pd.read_csv('Data_behaviour_per_day.csv')
feature_cols = df.columns[7:]
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n, m = X_scaled.shape
k = 4

# --- Reference solution ---
Z_ref, S_ref = archetypal_analysis(X_scaled, k, seed=42)

# --- Bootstrap: resample rows, re-fit, re-project each sample ---
# Since bootstrap resamples with replacement, some samples are duplicated,
# some missing. For each sample we track: for each bootstrap run, what are
# its coefficients in the fitted model.
n_boot = 200
boot_S = np.zeros((n_boot, n, k))  # [bootstrap_run, sample, archetype]

for b in range(n_boot):
    boot_idx = np.random.choice(n, n, replace=True)
    X_boot = X_scaled[boot_idx]
    try:
        Z_boot, S_boot = archetypal_analysis(X_boot, k, seed=b)
        # Now project ALL original samples onto these archetypes
        S_proj = np.zeros((n, k))
        for i in range(n):
            S_proj[i] = _solve_convex_coeff(Z_boot.T, X_scaled[i])
        boot_S[b] = S_proj
    except:
        boot_S[b] = np.nan

# Remove failed runs
boot_S = boot_S[~np.isnan(boot_S).any(axis=(1,2))]
print(f"Successful bootstrap runs: {len(boot_S)}")

# --- Compute stats ---
S_mean = boot_S.mean(axis=0)
S_std = boot_S.std(axis=0)
S_ci_low = np.percentile(boot_S, 2.5, axis=0)
S_ci_high = np.percentile(boot_S, 97.5, axis=0)

# Dominant archetype from reference
dominant_ref = np.argmax(S_ref, axis=1)
# Bootstrap probability of being dominant for each sample
dominant_boot = np.argmax(boot_S, axis=2)  # (n_boot x n)
dominant_prob = np.zeros((n, k))
for s in range(n):
    for a in range(k):
        dominant_prob[s, a] = (dominant_boot[:, s] == a).mean()

arch_names = ['Arch 1: Solitary\nHigh-Accel', 'Arch 2: Hiding/\nInactive',
              'Arch 3: Social\nExplorer', 'Arch 4: Social\nContact']

# ================================================================
# Print results
# ================================================================
print(f"\n{'='*100}")
print(f"{'Sample':<30} {'Sex':<6} {'Hier':<8} {'Dom':<8} {'     S coefficients (mean ± std)' :<60} {'Boot P(dom)':<12}")
print(f"{'='*100}")

for i in range(n):
    exp = df['Experiment'].iloc[i][:7]
    sex = df['sex'].iloc[i][:1].upper()
    hier = df['Hierarchy'].iloc[i]
    dom = dominant_ref[i]
    dom_name = f'Arch {dom+1}'
    
    coeff_str = ' | '.join([f'{S_ref[i,a]:.3f}' for a in range(k)])
    boot_str = ' | '.join([f'{S_mean[i,a]:.3f}±{S_std[i,a]:.3f}' for a in range(k)])
    p_str = f'{dominant_prob[i, dom]:.3f}'
    
    coords = f'({df["Mice.chips"].iloc[i]})'
    print(f'{exp:<10} {sex:<6} {hier:<8} {dom_name:<8}  S=[{coeff_str}]  boot=[{boot_str}]  P(dom)={p_str}')

# ================================================================
# Visualizations
# ================================================================

# 1. Probability matrix heatmap
fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(dominant_prob.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_yticks(range(k))
ax.set_yticklabels([f'Archetype {i+1}' for i in range(k)], fontsize=11)
ax.set_xticks(range(n))
ax.set_xticklabels([f'{df["Experiment"].iloc[i][:5]}-{df["Hierarchy"].iloc[i][:3]}({df["sex"].iloc[i][:1].upper()})' 
                     for i in range(n)], rotation=90, fontsize=6)
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Archetype', fontsize=12)
ax.set_title('Bootstrap probability of being assigned to each archetype', fontsize=14, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Probability', fontsize=11)
plt.tight_layout()
plt.savefig('bootstrap_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: bootstrap_probabilities.png")

# 2. Coefficient distributions for 4 representative samples
rep_indices = [0, 12, 30, 58]  # pick diverse samples
rep_labels = [f'{df["Experiment"].iloc[i]}-{df["Hierarchy"].iloc[i]}({df["sex"].iloc[i][:1].upper()})'
              for i in rep_indices]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
arch_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for idx, (si, ax) in enumerate(zip(rep_indices, axes)):
    for a in range(k):
        vals = boot_S[:, si, a]
        ax.hist(vals, bins=30, alpha=0.6, color=arch_colors[a],
                label=f'Arch {a+1}', density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(S_ref[si, 0], color=arch_colors[0], linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(S_ref[si, 1], color=arch_colors[1], linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(S_ref[si, 2], color=arch_colors[2], linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(S_ref[si, 3], color=arch_colors[3], linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Coefficient', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{rep_labels[idx]}  (dom=Arch {dominant_ref[si]+1})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Bootstrap distribution of archetype coefficients (4 example samples)\n'
             '(dashed lines = reference solution)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('bootstrap_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bootstrap_distributions.png")

# 3. Uncertainty summary: S_mean vs S_std, colored by confidence
fig, ax = plt.subplots(figsize=(10, 4))
# For each sample, plot its dominant archetype coefficient with 95% CI
xpos = np.arange(n)
for i in range(n):
    dom = dominant_ref[i]
    ax.errorbar(i, S_mean[i, dom], yerr=S_std[i, dom]*1.96,
                fmt='o', color=arch_colors[dom], alpha=0.7, capsize=2, markersize=5)
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Dominant archetype\ncoefficient', fontsize=12)
ax.set_title('Dominant archetype coefficient with 95% bootstrapped CI', fontsize=14, fontweight='bold')
ax.set_xticks(range(n))
ax.set_xticklabels([f'{df["Experiment"].iloc[i][:5]}-{df["Hierarchy"].iloc[i][:3]}'
                     for i in range(n)], rotation=90, fontsize=5)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('coefficient_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: coefficient_uncertainty.png")

# 4. Classification confidence: how many samples have P(dom) > threshold?
print("\n=== Classification confidence summary ===")
for thresh in [0.5, 0.7, 0.9, 0.95]:
    confident = (dominant_prob.max(axis=1) >= thresh).sum()
    print(f"  P(dominant) >= {thresh:.2f}: {confident}/{n} samples ({confident/n*100:.0f}%)")

# Per archetype
print("\nPer archetype confidence (P(dominant) >= 0.7):")
for a in range(k):
    in_arch = dominant_ref == a
    if in_arch.sum() > 0:
        confident = (dominant_prob[in_arch, a] >= 0.7).sum()
        print(f"  Archetype {a+1} ({in_arch.sum()} samples): {confident}/{in_arch.sum()} ({confident/in_arch.sum()*100:.0f}%)")
