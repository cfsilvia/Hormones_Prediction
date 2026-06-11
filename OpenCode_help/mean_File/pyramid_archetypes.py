import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
scaler = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
# Actually just redo it properly
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 4
Z, S = archetypal_analysis(X_scaled, k)
Z_original = scaler.inverse_transform(Z)

df['Dominant'] = pd.DataFrame(S, columns=['A1','A2','A3','A4']).idxmax(axis=1)
arch_names = {
    'A1': 'Solitary\nHigh-Acceleration',
    'A2': 'Hiding/\nInactive',
    'A3': 'Social\nExplorer',
    'A4': 'Social\nContact'
}
arch_colors = {'A1': '#e74c3c', 'A2': '#3498db', 'A3': '#2ecc71', 'A4': '#9b59b6'}
arch_short = ['A1','A2','A3','A4']

# ---- Pyramid 1: Population pyramid by Sex ----
fig, ax = plt.subplots(figsize=(10, 7))

male_counts = []
female_counts = []
for a in arch_short:
    male_counts.append(((df['sex'] == 'male') & (df['Dominant'] == a)).sum())
    female_counts.append(((df['sex'] == 'female') & (df['Dominant'] == a)).sum())

max_count = max(max(male_counts), max(female_counts)) + 1
y_pos = np.arange(len(arch_short))
bar_height = 0.6

for i in range(len(arch_short)):
    ax.barh(y_pos[i], -male_counts[i], bar_height, color=arch_colors[arch_short[i]], alpha=0.85)
    ax.barh(y_pos[i], female_counts[i], bar_height, color=arch_colors[arch_short[i]], alpha=0.85)
    ax.text(-male_counts[i]-0.15, y_pos[i], str(male_counts[i]),
            ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(female_counts[i]+0.15, y_pos[i], str(female_counts[i]),
            ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(0, y_pos[i], arch_names[arch_short[i]],
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=arch_colors[arch_short[i]], alpha=0.9))

ax.set_yticks(y_pos)
ax.set_yticklabels(['']*len(arch_short))
ax.set_xlim(-max_count-1, max_count+1)
ax.set_xticks(range(-max_count, max_count+1))
ax.set_xticklabels([str(abs(x)) for x in range(-max_count, max_count+1)])
ax.axvline(0, color='black', linewidth=1.5)
ax.set_xlabel('Number of mice', fontsize=13)
ax.set_title('Archetype Distribution by Sex', fontsize=16, fontweight='bold', pad=15)
ax.text(-max_count*0.5-0.5, -0.5, 'Male', ha='center', fontsize=13, fontweight='bold')
ax.text(max_count*0.5+0.5, -0.5, 'Female', ha='center', fontsize=13, fontweight='bold')
ax.text(-max_count-0.8, -0.8, f'n={sum(male_counts)} mice', fontsize=10, color='#555')
ax.text(max_count+0.8, -0.8, f'n={sum(female_counts)} mice', fontsize=10, color='#555')

plt.tight_layout()
plt.savefig('pyramid_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Pyramid 2: Archetype composition across Experiments (stacked) ----
fig, ax = plt.subplots(figsize=(12, 7))
experiments = sorted(df['Experiment'].unique())
bottom = np.zeros(len(experiments))
for a in arch_short:
    vals = []
    for e in experiments:
        mask = df['Experiment'] == e
        vals.append((df.loc[mask, 'Dominant'] == a).mean() * 100)
    ax.bar(experiments, vals, bottom=bottom, label=arch_names[a],
           color=arch_colors[a], alpha=0.85, width=0.7, edgecolor='white', linewidth=0.5)
    bottom += vals

ax.set_ylabel('Percentage (%)', fontsize=13)
ax.set_xlabel('Experiment', fontsize=13)
ax.set_title('Archetype Composition by Experiment Cohort', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('pyramid_by_experiment.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Pyramid 3: Mean archetype coefficients by Hierarchy -(pyramid style) ----
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
hierarchies = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
hier_colors = ['#1abc9c', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']

for idx, a in enumerate(arch_short):
    ax = axes[idx]
    left_vals = []
    right_vals = []
    for h in hierarchies:
        mask = (df['Hierarchy'] == h) & (df['sex'] == 'male')
        left_vals.append(S[mask, idx].mean() if mask.sum() > 0 else 0)
        mask = (df['Hierarchy'] == h) & (df['sex'] == 'female')
        right_vals.append(S[mask, idx].mean() if mask.sum() > 0 else 0)

    y_pos = np.arange(len(hierarchies))
    for i in range(len(hierarchies)):
        ax.barh(y_pos[i], -left_vals[i], 0.5, color=hier_colors[i], alpha=0.8)
        ax.barh(y_pos[i], right_vals[i], 0.5, color=hier_colors[i], alpha=0.8)
        ax.text(-left_vals[i]-0.01, y_pos[i], f'{left_vals[i]:.2f}',
                ha='right', va='center', fontsize=8)
        ax.text(right_vals[i]+0.01, y_pos[i], f'{right_vals[i]:.2f}',
                ha='left', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hierarchies if idx == 0 else [])
    ax.set_xlim(-0.6, 0.6)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(arch_names[a], fontsize=10, fontweight='bold')
    if idx == 0:
        ax.text(-0.55, -0.5, 'Male', ha='center', fontsize=9, fontweight='bold')
        ax.text(0.55, -0.5, 'Female', ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Mean Archetype Coefficient by Hierarchy and Sex', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pyramid_hierarchy_sex.png', dpi=150, bbox_inches='tight')
plt.close()

print("Pyramid visualizations saved:")
print("  pyramid_by_sex.png")
print("  pyramid_by_experiment.png")
print("  pyramid_hierarchy_sex.png")
