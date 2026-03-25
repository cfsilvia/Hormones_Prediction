import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class PCHA:
    def __init__(self, k=4, n_iter=100, normalize=True, random_state=0):
        self.k = k
        self.n_iter = n_iter
        self.normalize = normalize
        self.rng = np.random.default_rng(random_state)

    # ======================
    # Simplex projection
    # ======================
    def _project_simplex(self, v):
        v = np.maximum(v, 0)
        if v.sum() == 0:
            return np.ones_like(v) / len(v)
        return v / v.sum()

    # ======================
    # Core PCHA
    # ======================
    def _fit_pcha(self, X):
        n, d = X.shape

        # Initialization
        idx = self.rng.choice(n, self.k, replace=False)
        B = np.zeros((n, self.k))
        B[idx, np.arange(self.k)] = 1

        A = self.rng.random((n, self.k))
        A /= A.sum(axis=1, keepdims=True)

        for _ in range(self.n_iter):

            Z = B.T @ X  # archetypes

            # ---- Update A ----
            cons_A = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds_A = [(0, 1)] * self.k

            for i in range(n):
                def loss(a):
                    return np.linalg.norm(X[i] - a @ Z)**2

                res = minimize(loss, A[i], method='SLSQP',
                               bounds=bounds_A, constraints=cons_A)
                A[i] = self._project_simplex(res.x)

            # ---- Update B ----
            cons_B = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds_B = [(0, 1)] * n

            for j in range(self.k):
                def loss(b):
                    Z_temp = Z.copy()
                    Z_temp[j] = b @ X
                    recon = A @ Z_temp
                    return np.linalg.norm(X - recon)**2

                res = minimize(loss, B[:, j], method='SLSQP',
                               bounds=bounds_B, constraints=cons_B)
                B[:, j] = self._project_simplex(res.x)

        Z = B.T @ X
        recon = A @ Z

        point_errors = np.linalg.norm(X - recon, axis=1)
        mean_error = point_errors.mean()

        return Z, A, B, mean_error, point_errors

    # ======================
    # P-VALUE ESTIMATION
    # ======================
    def _compute_pvalues(self, X, n_perm=100):
        """
        Compute p-values by comparing reconstruction error
        to random permutations (null model)
        """
        n = X.shape[0]
        null_errors = np.zeros((n_perm, n))

        for p in range(n_perm):
            X_perm = X.copy()
            for col in range(X.shape[1]):
                self.rng.shuffle(X_perm[:, col])

            _, _, _, _, err = self._fit_pcha(X_perm)
            null_errors[p] = err

        # empirical p-value
        pvals = np.mean(null_errors >= self.point_errors, axis=0)

        return pvals

    # ======================
    # Fit
    # ======================
    def fit(self, X_raw, compute_pvalues=True):

        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X_raw)
        else:
            X = X_raw.copy()

        self.X = X

        self.Z, self.A, self.B, self.error, self.point_errors = self._fit_pcha(X)

        print("Mean reconstruction error:", self.error)

        if compute_pvalues:
            print("Computing p-values (this may take time)...")
            self.pvalues = self._compute_pvalues(X)
        else:
            self.pvalues = None

        return self

    # ======================
    # SAVE TO EXCEL
    # ======================
    def save_to_excel(self, filename="pcha_results.xlsx"):

        # ---- Points ----
        df_points = pd.DataFrame(self.X, columns=[f"dim_{i}" for i in range(self.X.shape[1])])
        df_points["reconstruction_error"] = self.point_errors

        if self.pvalues is not None:
            df_points["p_value"] = self.pvalues

        # ---- Archetypes ----
        df_archetypes = pd.DataFrame(self.Z, columns=[f"dim_{i}" for i in range(self.Z.shape[1])])

        with pd.ExcelWriter(filename) as writer:
            df_points.to_excel(writer, sheet_name="points", index=False)
            df_archetypes.to_excel(writer, sheet_name="archetypes", index=False)

        print(f"Saved results to {filename}")

    # ======================
    # 3D Plot
    # ======================
    def plot_3d(self):

        X_3d = self.X[:, :3]
        Z_3d = self.Z[:, :3]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], alpha=0.6)
        ax.scatter(Z_3d[:,0], Z_3d[:,1], Z_3d[:,2],
                   marker='^', s=200)

        for i in range(self.k):
            for j in range(i+1, self.k):
                ax.plot([Z_3d[i,0], Z_3d[j,0]],
                        [Z_3d[i,1], Z_3d[j,1]],
                        [Z_3d[i,2], Z_3d[j,2]])

        ax.set_title("PCHA Simplex")
        plt.show()