import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def _solve_convex_coeff(A, b):
    """Solve min ||A @ x - b||^2 s.t. x >= 0, sum(x) = 1"""
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
    """
    Archetypal analysis using alternating convex least squares.
    X: (n, m) array
    k: number of archetypes
    """
    np.random.seed(42)
    n, m = X.shape
    
    # Initialize: pick k random data points
    idx = np.random.choice(n, k, replace=False)
    Z = X[idx].copy()
    
    for iteration in range(max_iter):
        Z_old = Z.copy()
        
        # S-step: X ≈ S @ Z, S rows are convex coefficients
        S = np.zeros((n, k))
        for i in range(n):
            S[i] = _solve_convex_coeff(Z.T, X[i])
        
        # C-step: Z = C @ X, C rows are convex coefficients
        C = np.zeros((k, n))
        for j in range(k):
            C[j] = _solve_convex_coeff(X.T, Z[j])
        
        Z = C @ X
        
        diff = np.linalg.norm(Z - Z_old) / max(1e-8, np.linalg.norm(Z_old))
        if diff < tol:
            break
    
    return Z, S


def compute_sse(X, Z, S):
    return np.sum((X - S @ Z) ** 2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
df = pd.read_csv('Data_behaviour_per_day.csv')

feature_cols = df.columns[7:]
X = df[feature_cols].values
col_names = feature_cols.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples, n_features = X_scaled.shape
print(f"Data shape: {n_samples} samples, {n_features} features")
print()

# Run for k=4 archetypes
k = 4
print(f"Finding k={k} archetypes...")
Z, S = archetypal_analysis(X_scaled, k)
sse = compute_sse(X_scaled, Z, S)
print(f"Final SSE: {sse:.4f}")

Z_original = scaler.inverse_transform(Z)

print()
print("=== Archetypes ===")
for i in range(k):
    print(f"\n--- Archetype {i+1} ---")
    z_scores = (Z_original[i] - scaler.mean_) / np.sqrt(scaler.var_)
    top_pos = np.argsort(z_scores)[-5:][::-1]
    top_neg = np.argsort(z_scores)[:5]
    print("  Highest:")
    for idx in top_pos:
        print(f"    {col_names[idx]}: {Z_original[i][idx]:.4f} (z={z_scores[idx]:.2f})")
    print("  Lowest:")
    for idx in top_neg:
        print(f"    {col_names[idx]}: {Z_original[i][idx]:.4f} (z={z_scores[idx]:.2f})")

print()
print("=== Sample archetype coefficients ===")
df_coeff = pd.DataFrame(S, columns=[f'Archetype_{i+1}' for i in range(k)])
df_coeff = df_coeff.round(4)
df_result = df[['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']].copy()
df_result = pd.concat([df_result, df_coeff], axis=1)
df_result['Dominant'] = df_coeff.idxmax(axis=1)
print(df_result.to_string(index=False))

print()
print("=== Dominant archetype by Hierarchy ===")
ct = pd.crosstab(df_result['Hierarchy'], df_result['Dominant'])
print(ct.to_string())

print()
print("=== Dominant archetype by Sex ===")
ct2 = pd.crosstab(df_result['sex'], df_result['Dominant'])
print(ct2.to_string())
