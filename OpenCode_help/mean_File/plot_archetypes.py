import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
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
Z, S = archetypal_analysis(X_scaled, k)
Z_original = scaler.inverse_transform(Z)

# Plot 1: Archetype profiles as heatmap
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i in range(k):
    z_scores = (Z_original[i] - scaler.mean_) / np.sqrt(scaler.var_)
    ax = axes[i]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in z_scores]
    ax.barh(range(len(col_names)), z_scores, color=colors)
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names, fontsize=7)
    ax.set_xlabel('Z-score')
    ax.set_title(f'Archetype {i+1}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig('archetypes_profiles.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Archetype composition by Hierarchy
fig, ax = plt.subplots(figsize=(10, 6))
hierarchies = df['Hierarchy'].unique()
bar_width = 0.2
x = np.arange(len(hierarchies))
for i in range(k):
    means = []
    for h in hierarchies:
        mask = df['Hierarchy'] == h
        means.append(S[mask, i].mean())
    ax.bar(x + i * bar_width, means, bar_width, label=f'Archetype {i+1}')
ax.set_xlabel('Hierarchy', fontsize=12)
ax.set_ylabel('Mean coefficient', fontsize=12)
ax.set_title('Archetype composition by Hierarchy', fontsize=14)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(hierarchies)
ax.legend()
plt.tight_layout()
plt.savefig('archetypes_by_hierarchy.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Archetype composition by Sex
fig, ax = plt.subplots(figsize=(8, 5))
sexes = ['male', 'female']
x = np.arange(len(sexes))
for i in range(k):
    means = []
    for s in sexes:
        mask = df['sex'] == s
        means.append(S[mask, i].mean())
    ax.bar(x + i * bar_width, means, bar_width, label=f'Archetype {i+1}')
ax.set_xlabel('Sex', fontsize=12)
ax.set_ylabel('Mean coefficient', fontsize=12)
ax.set_title('Archetype composition by Sex', fontsize=14)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(sexes)
ax.legend()
plt.tight_layout()
plt.savefig('archetypes_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved: archetypes_profiles.png, archetypes_by_hierarchy.png, archetypes_by_sex.png")
