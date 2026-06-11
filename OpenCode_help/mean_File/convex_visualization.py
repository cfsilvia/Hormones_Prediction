import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
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

def archetypal_analysis(X, k, max_iter=200, tol=1e-6):
    np.random.seed(42)
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

k = 4
Z_scaled, S = archetypal_analysis(X_scaled, k)
Z_original = scaler.inverse_transform(Z_scaled)

# PCA to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Z_pca = pca.transform(Z_scaled)

var_explained = pca.explained_variance_ratio_ * 100

# Map each sample to its closest archetype
dominant = np.argmax(S, axis=1)

arch_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
arch_labels = ['Arch 1: Solitary High-Accel', 'Arch 2: Hiding/Inactive',
               'Arch 3: Social Explorer', 'Arch 4: Social Contact']
sex_markers = {'male': 'o', 'female': '^'}

fig, ax = plt.subplots(figsize=(12, 10))

# Plot all data points colored by dominant archetype, shaped by sex
for i in range(k):
    mask = dominant == i
    for sex in ['male', 'female']:
        m = mask & (df['sex'] == sex)
        ax.scatter(X_pca[m, 0], X_pca[m, 1],
                   c=[arch_colors[i]], marker=sex_markers[sex],
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'{arch_labels[i]} - {sex}' if i == 0 else '')

# Plot archetypes
for i in range(k):
    ax.scatter(Z_pca[i, 0], Z_pca[i, 1], c=[arch_colors[i]],
               marker='D', s=350, edgecolors='black', linewidth=2, zorder=5)
    ax.annotate(f'Archetype {i+1}', (Z_pca[i, 0], Z_pca[i, 1]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=11, fontweight='bold', color=arch_colors[i])

# Draw convex hull of archetypes
hull = ConvexHull(Z_pca)
for simplex in hull.simplices:
    ax.plot(Z_pca[simplex, 0], Z_pca[simplex, 1], 'k--', linewidth=2, alpha=0.6)

# Also draw convex hull of all data points (outer boundary)
data_hull = ConvexHull(X_pca)
for simplex in data_hull.simplices:
    ax.plot(X_pca[simplex, 0], X_pca[simplex, 1], 'gray', linewidth=1, alpha=0.3)

ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)', fontsize=13)
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)', fontsize=13)
ax.set_title('Archetypal Analysis: Archetypes form convex hull around data\n'
             '(dashed = archetype convex hull, gray = data convex hull)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convex_hull_archetypes.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Second plot: PAIRWISE ternary-style with all coefficients ----
# Show each data point colored by its mixture weights (RGB for 3 archetypes)
# For 4 archetypes, use RGB where the 4th is represented by brightness
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

for idx, (a, b) in enumerate(pairs):
    ax = axes[idx]
    # Evenly distribute remaining weight color
    colors_pt = []
    for j in range(len(S)):
        r = float(S[j, a])
        g = float(S[j, b])
        other = float(max(0, 1 - r - g))
        colors_pt.append((min(1.0, max(0.0, r)), min(1.0, max(0.0, g)), min(1.0, max(0.0, other))))

    scatter = ax.scatter(S[:, a], S[:, b], c=colors_pt, s=80, alpha=0.8,
                         edgecolors='black', linewidth=0.5)
    # Label archetype positions
    corners = [(1,0,0), (0,1,0), (0,0,1)]
    arch_positions = [(1,0), (0,1), (0,0)]
    ax.scatter([1, 0, 0], [0, 1, 0], c=['red','green','blue'], s=200,
               marker='D', edgecolors='black', linewidth=2, zorder=5)
    for label, (x, y) in zip(['Arch 1', 'Arch 2', 'Mixed'],
                              [(1.05, 0), (-0.05, 1.05), (-0.05, -0.05)]):
        ax.text(x, y, label, fontsize=9, fontweight='bold')

    ax.set_xlabel(f'Archetype {a+1} coeff', fontsize=11)
    ax.set_ylabel(f'Archetype {b+1} coeff', fontsize=11)
    ax.set_title(f'Archetypes {a+1} vs {b+1}', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.suptitle('Pairwise Archetype Coefficient Spaces\n(Pure archetypes at corners, data points inside)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pairwise_archetype_space.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Third plot: simplex weights as stacked bars showing convex combos ----
fig, ax = plt.subplots(figsize=(14, 6))
xpos = np.arange(len(S))
bottom = np.zeros(len(S))
for i in range(k):
    ax.bar(xpos, S[:, i], bottom=bottom, label=arch_labels[i],
           color=arch_colors[i], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.3)
    bottom += S[:, i]

ax.set_xlabel('Samples (sorted by dominant archetype)', fontsize=12)
ax.set_ylabel('Archetype coefficient', fontsize=12)
ax.set_title('Each sample = convex combination of 4 archetypes (bars sum to 1)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-0.5, len(S) - 0.5)
ax.set_ylim(0, 1.05)

# Sort indices by dominant archetype for cleaner display
sort_idx = np.argsort(dominant)
xtick_labels = []
for idx in sort_idx:
    arch = dominant[idx] + 1
    exp = df['Experiment'].iloc[idx]
    hier = df['Hierarchy'].iloc[idx]
    sex_abbr = 'M' if df['sex'].iloc[idx] == 'male' else 'F'
    xtick_labels.append(f'{exp}-{hier}({sex_abbr})')
ax.set_xticks(xpos)
ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig('convex_combination_bars.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: convex_hull_archetypes.png")
print("Saved: pairwise_archetype_space.png")
print("Saved: convex_combination_bars.png")
