import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def _solve_convex_coeff(A, b):
    n = A.shape[1]
    x0 = np.ones(n) / n
    def loss(x): return np.sum((A @ x - b) ** 2)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1)] * n
    res = minimize(loss, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
    return res.x

def AA(X, k, seed=42):
    np.random.seed(seed)
    n, m = X.shape
    idx = np.random.choice(n, k, replace=False)
    Z = X[idx].copy()
    for _ in range(200):
        Zo = Z.copy()
        S = np.zeros((n, k))
        for i in range(n):
            S[i] = _solve_convex_coeff(Z.T, X[i])
        C = np.zeros((k, n))
        for j in range(k):
            C[j] = _solve_convex_coeff(X.T, Z[j])
        Z = C @ X
        if np.linalg.norm(Z - Zo) < 1e-6: break
    return Z, S

df = pd.read_csv('Data_behaviour_per_day.csv')
cols = df.columns[7:]
X = df[cols].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Z, S = AA(Xs, 4)
Zo = scaler.inverse_transform(Z)

chase_cols = [
    'Chasing.(N.events)', 'Chasing.duration.(sec)',
    'Being.chased.(N.events)', 'Being.chased.duration.(sec)',
    'Chasing.plus.being.chasing.all(N.events)', 'Chasing.plus.being.chasing.duration(sec)'
]
chase_idx = [list(cols).index(c) for c in chase_cols]

print(f'Chasing features z-scores across archetypes:')
print(f'{"Feature":<50} {"Arch1":>8} {"Arch2":>8} {"Arch3":>8} {"Arch4":>8}')
print('-'*82)
for c, idx in zip(chase_cols, chase_idx):
    zs = [(Zo[i, idx] - scaler.mean_[idx]) / np.sqrt(scaler.var_[idx]) for i in range(4)]
    print(f'{c:<50}', end='')
    for z in zs:
        print(f'{z:>8.2f}', end='')
    print()

# Top 15 features by max |z| across archetypes
print('\nTop 15 discriminating features (by max |z| across archetypes):')
all_z = []
for i in range(54):
    zs = [(Zo[a,i]-scaler.mean_[i])/np.sqrt(scaler.var_[i]) for a in range(4)]
    max_abs_z = max(abs(z) for z in zs)
    all_z.append((max_abs_z, cols[i], zs))
all_z.sort(reverse=True)
print(f'{"Rank":<5} {"Feature":<50} {"A1z":>7} {"A2z":>7} {"A3z":>7} {"A4z":>7}')
for r, (mz, name, zs) in enumerate(all_z[:15], 1):
    print(f'{r:<5} {name:<50}', end='')
    for z in zs:
        print(f'{z:>7.2f}', end='')
    print()
