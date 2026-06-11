import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 4
Z_scaled, S = archetypal_analysis(X_scaled, k)

pca = PCA(n_components=3)
X_pca3 = pca.fit_transform(X_scaled)
Z_pca3 = pca.transform(Z_scaled)

var = pca.explained_variance_ratio_ * 100

dominant = np.argmax(S, axis=1)
arch_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
arch_labels = ['Arch 1: Solitary\nHigh-Accel', 'Arch 2: Hiding/\nInactive',
               'Arch 3: Social\nExplorer', 'Arch 4: Social\nContact']

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

for i in range(k):
    mask = dominant == i
    for sex, marker, label in [('male', 'o', 'male'), ('female', '^', 'female')]:
        m = mask & (df['sex'] == sex)
        if m.sum() == 0:
            continue
        ax.scatter(X_pca3[m, 0], X_pca3[m, 1], X_pca3[m, 2],
                   c=[arch_colors[i]], marker=marker, s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=f'{arch_labels[i]} - {label}' if i == 0 else '')

# Plot archetypes as large labeled spheres
for i in range(k):
    ax.scatter(Z_pca3[i, 0], Z_pca3[i, 1], Z_pca3[i, 2],
               c=[arch_colors[i]], marker='D', s=500, edgecolors='black', linewidth=2, zorder=10)
    ax.text(Z_pca3[i, 0], Z_pca3[i, 1], Z_pca3[i, 2],
            f'  Arch {i+1}', fontsize=13, fontweight='bold', color='black')

# Draw convex hull edges between archetypes
from scipy.spatial import ConvexHull
hull = ConvexHull(Z_pca3)
for simplex in hull.simplices:
    pts = Z_pca3[simplex]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k--', linewidth=1.5, alpha=0.5)

ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=12, labelpad=10)
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=12, labelpad=10)
ax.set_zlabel(f'PC3 ({var[2]:.1f}%)', fontsize=12, labelpad=10)
ax.set_title('Archetypes in 3D PCA Space\n(archetypes = diamonds, dashed = convex hull)',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.view_init(elev=25, azim=-45)
plt.tight_layout()
plt.savefig('archetypes_3d.png', dpi=150, bbox_inches='tight')
plt.close()

# Second view: different angle
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

for i in range(k):
    mask = dominant == i
    for sex, marker in [('male', 'o'), ('female', '^')]:
        m = mask & (df['sex'] == sex)
        if m.sum() == 0:
            continue
        ax.scatter(X_pca3[m, 0], X_pca3[m, 1], X_pca3[m, 2],
                   c=[arch_colors[i]], marker=marker, s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

for i in range(k):
    ax.scatter(Z_pca3[i, 0], Z_pca3[i, 1], Z_pca3[i, 2],
               c=[arch_colors[i]], marker='D', s=500, edgecolors='black', linewidth=2, zorder=10)
    ax.text(Z_pca3[i, 0], Z_pca3[i, 1], Z_pca3[i, 2],
            f'  Arch {i+1}', fontsize=13, fontweight='bold', color='black')

for simplex in hull.simplices:
    pts = Z_pca3[simplex]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k--', linewidth=1.5, alpha=0.5)

ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=12, labelpad=10)
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=12, labelpad=10)
ax.set_zlabel(f'PC3 ({var[2]:.1f}%)', fontsize=12, labelpad=10)
ax.set_title('Archetypes in 3D PCA Space (alternate view)',
             fontsize=15, fontweight='bold', pad=20)
ax.view_init(elev=15, azim=135)
plt.tight_layout()
plt.savefig('archetypes_3d_alt.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: archetypes_3d.png")
print("Saved: archetypes_3d_alt.png")
