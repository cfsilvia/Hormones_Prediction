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

# --- Load all data (including outlier) ---
df = pd.read_csv('Data_behaviour_per_day.csv')
print(f"Total rows: {len(df)}")

feature_cols = df.columns[7:]
X = df[feature_cols].values
col_names = feature_cols.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_samples, n_features = X_scaled.shape

k = 3
Z_scaled, S = archetypal_analysis(X_scaled, k)
sse = np.sum((X_scaled - S @ Z_scaled) ** 2)
print(f"k=3 SSE: {sse:.4f}")

Z_original = scaler.inverse_transform(Z_scaled)

arch_colors = ['#e74c3c', '#2ecc71', '#3498db']
arch_labels = ['Arch 1', 'Arch 2', 'Arch 3']

print("\n=== Archetypes (k=3, all data) ===")
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

df_coeff = pd.DataFrame(S, columns=arch_labels).round(4)
result = df[['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips','Animal']].copy()
result = pd.concat([result, df_coeff], axis=1)
result['Dominant'] = df_coeff.idxmax(axis=1)

print("\n=== Coefficients ===")
print(result.to_string(index=False))
print("\n=== By Hierarchy ===")
print(pd.crosstab(result['Hierarchy'], result['Dominant']).to_string())
print("\n=== By Sex ===")
print(pd.crosstab(result['sex'], result['Dominant']).to_string())

# === VISUALIZATIONS ===

# 1. Pyramid by sex
fig, ax = plt.subplots(figsize=(10, 6))
arch_names_viz = ['Arch 1', 'Arch 2', 'Arch 3']
male_counts = [((df['sex']=='male')&(result['Dominant']==a)).sum() for a in arch_labels]
female_counts = [((df['sex']=='female')&(result['Dominant']==a)).sum() for a in arch_labels]
maxc = max(max(male_counts), max(female_counts)) + 1
y_pos = np.arange(k)
for i in range(k):
    ax.barh(y_pos[i], -male_counts[i], 0.6, color=arch_colors[i], alpha=0.85)
    ax.barh(y_pos[i], female_counts[i], 0.6, color=arch_colors[i], alpha=0.85)
    ax.text(-male_counts[i]-0.15, y_pos[i], str(male_counts[i]), ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(female_counts[i]+0.15, y_pos[i], str(female_counts[i]), ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(0, y_pos[i], arch_names_viz[i], ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=arch_colors[i], alpha=0.9))
ax.set_yticks(y_pos)
ax.set_yticklabels(['']*k)
ax.set_xlim(-maxc-1, maxc+1)
ax.set_xticks(range(-maxc, maxc+1))
ax.set_xticklabels([str(abs(x)) for x in range(-maxc, maxc+1)])
ax.axvline(0, color='black', linewidth=1.5)
ax.set_xlabel('Number of mice', fontsize=13)
ax.set_title('Archetype Distribution by Sex (k=3, all data)', fontsize=16, fontweight='bold', pad=15)
ax.text(-maxc*0.5-0.5, -0.5, 'Male', ha='center', fontsize=13, fontweight='bold')
ax.text(maxc*0.5+0.5, -0.5, 'Female', ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pyramid_by_sex_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pyramid_by_sex_k3.png")

# 2. 3D PCA
pca = PCA(n_components=3)
X3 = pca.fit_transform(X_scaled)
Z3 = pca.transform(Z_scaled)
var = pca.explained_variance_ratio_ * 100
dominant = result['Dominant'].values

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')
for i in range(k):
    mask_arr = dominant == arch_labels[i]
    for sex, marker in [('male', 'o'), ('female', '^')]:
        m = mask_arr & (df['sex'] == sex)
        if m.sum() == 0: continue
        ax.scatter(X3[m,0], X3[m,1], X3[m,2], c=[arch_colors[i]], marker=marker,
                   s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
for i in range(k):
    ax.scatter(Z3[i,0], Z3[i,1], Z3[i,2], c=[arch_colors[i]], marker='D',
               s=500, edgecolors='black', linewidth=2, zorder=10)
    ax.text(Z3[i,0], Z3[i,1], Z3[i,2], f'  {arch_labels[i]}', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=12)
ax.set_zlabel(f'PC3 ({var[2]:.1f}%)', fontsize=12)
ax.set_title('k=3 Archetypes in 3D PCA (all data)', fontsize=15, fontweight='bold')
ax.view_init(elev=25, azim=-45)
plt.tight_layout()
plt.savefig('archetypes_3d_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: archetypes_3d_k3.png")

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')
for i in range(k):
    mask_arr = dominant == arch_labels[i]
    for sex, marker in [('male', 'o'), ('female', '^')]:
        m = mask_arr & (df['sex'] == sex)
        if m.sum() == 0: continue
        ax.scatter(X3[m,0], X3[m,1], X3[m,2], c=[arch_colors[i]], marker=marker,
                   s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
for i in range(k):
    ax.scatter(Z3[i,0], Z3[i,1], Z3[i,2], c=[arch_colors[i]], marker='D',
               s=500, edgecolors='black', linewidth=2, zorder=10)
    ax.text(Z3[i,0], Z3[i,1], Z3[i,2], f'  {arch_labels[i]}', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=12)
ax.set_zlabel(f'PC3 ({var[2]:.1f}%)', fontsize=12)
ax.set_title('k=3 Archetypes in 3D PCA - alt view (all data)', fontsize=15, fontweight='bold')
ax.view_init(elev=15, azim=135)
plt.tight_layout()
plt.savefig('archetypes_3d_alt_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: archetypes_3d_alt_k3.png")

# 3. Convex hull check (2D since 3 points form a triangle)
from scipy.spatial import Delaunay as Delaunay2D
pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(X_scaled)
Z2 = pca2.transform(Z_scaled)
hull_d = Delaunay2D(Z2)
inside = hull_d.find_simplex(X2) >= 0
print(f"\nPoints inside archetype hull in 2D: {inside.sum()}/{len(inside)}")

# 4. Profile plots
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
for i in range(k):
    z_scores = (Z_original[i] - scaler.mean_) / np.sqrt(scaler.var_)
    ax = axes[i]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in z_scores]
    ax.barh(range(len(col_names)), z_scores, color=colors)
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names, fontsize=7)
    ax.set_xlabel('Z-score')
    ax.set_title(f'{arch_labels[i]}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig('archetypes_profiles_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: archetypes_profiles_k3.png")

# 5. Convex combination bars
sort_idx = np.argsort([arch_labels.index(d) for d in dominant])
fig, ax = plt.subplots(figsize=(14, 6))
xpos = np.arange(len(S))
bottom = np.zeros(len(S))
for i in range(k):
    ax.bar(xpos, S[sort_idx, i], bottom=bottom, label=arch_labels[i],
           color=arch_colors[i], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.3)
    bottom += S[sort_idx, i]
ax.set_xlabel('Samples (sorted by dominant archetype)', fontsize=12)
ax.set_ylabel('Archetype coefficient', fontsize=12)
ax.set_title('Each sample = convex combination of 3 archetypes (all data)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylim(0, 1.05)
labels = []
for idx in sort_idx:
    e = df['Experiment'].iloc[idx]
    h = df['Hierarchy'].iloc[idx]
    s = 'M' if df['sex'].iloc[idx]=='male' else 'F'
    labels.append(f'{e}-{h}({s})')
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=90, fontsize=5)
plt.tight_layout()
plt.savefig('convex_combination_bars_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: convex_combination_bars_k3.png")

# 6. Ternary plot (Arch 1 vs Arch 2 vs Arch 3)
fig, ax = plt.subplots(figsize=(10, 9))
# Map 3 coefficients to ternary coordinates
# Use simplex: x = 0.5 * (2*s2 + s3) / (s1+s2+s3), y = sqrt(3)/2 * s3 / (s1+s2+s3)
s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]
tot = s1 + s2 + s3
x_tern = 0.5 * (2*s2 + s3) / tot
y_tern = np.sqrt(3)/2 * s3 / tot

for i in range(k):
    mask_arr = dominant == arch_labels[i]
    for sex, marker in [('male', 'o'), ('female', '^')]:
        m = mask_arr & (df['sex'] == sex)
        if m.sum() == 0: continue
        ax.scatter(x_tern[m], y_tern[m], c=[arch_colors[i]], marker=marker,
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)
# Triangle vertices
tri_x = [0, 1, 0.5, 0]
tri_y = [0, 0, np.sqrt(3)/2, 0]
ax.plot(tri_x, tri_y, 'k-', linewidth=1.5, alpha=0.5)
ax.text(0, -0.03, arch_labels[0], ha='center', fontsize=12, fontweight='bold')
ax.text(1, -0.03, arch_labels[1], ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, np.sqrt(3)/2+0.03, arch_labels[2], ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, -0.08, f'Males={sum(male_counts)}, Females={sum(female_counts)}', ha='center', fontsize=10, color='gray')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.1, np.sqrt(3)/2+0.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Ternary plot of archetype coefficients (k=3, all data)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ternary_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ternary_k3.png")
