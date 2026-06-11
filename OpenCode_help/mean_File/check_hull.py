import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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
Xs = scaler.fit_transform(X)
Z, S = archetypal_analysis(Xs, 4)

print('S sums to 1:', np.allclose(S.sum(axis=1), 1, atol=1e-4))

X_recon = S @ Z
errors = np.linalg.norm(Xs - X_recon, axis=1)
print(f'Recon error per point: mean={errors.mean():.4f}, max={errors.max():.4f}')

pca = PCA(n_components=3)
X3 = pca.fit_transform(Xs)
Z3 = pca.transform(Z)
print(f'PCA 3D variance: {pca.explained_variance_ratio_.sum()*100:.1f}%')

hull = Delaunay(Z3)
inside = hull.find_simplex(X3) >= 0
print(f'Points inside archetype hull in 3D: {inside.sum()}/{len(inside)}')
outside = np.where(~inside)[0]
if len(outside) > 0:
    print('Outside points:')
    for i in outside:
        row = df.iloc[i]
        print(f'  {row["Experiment"]} {row["sex"]} {row["Hierarchy"]}')

Xr3 = pca.transform(X_recon)
inside_r = hull.find_simplex(Xr3) >= 0
print(f'Reconstructed points inside hull in 3D: {inside_r.sum()}/{len(inside_r)}')
