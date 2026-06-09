import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


class ModelEvaluator:

    def __init__(self, random_state=42):
        self.random_state = random_state

    # ====================================================
    # SINGLE MODEL METRICS
    # ====================================================

    def bic(self, X):
        return self.model.bic(X)

    def aic(self, X):
        return self.model.aic(X)

    def entropy(self, probabilities):

        eps = 1e-12

        entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)

        return entropy.mean()

    def silhouette(self, X, labels):

        return silhouette_score(X, labels)    
    
    # ====================================================
    # FIND OPTIMAL NUMBER OF ARCHETYPES
    # ====================================================
    def find_optimal_archetypes(self, X, k_range=range(2, 8), covariance_type="full"):
        results = []
        for k in k_range:
              model = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=self.random_state,
                n_init=20
            )
              model.fit(X)
              probabilities = model.predict_proba(X)
              labels = model.predict(X)
              bic_score = model.bic(X)
              aic_score = model.aic(X)
              entropy_score = self.entropy(probabilities)
              results.append({
                "n_archetypes": k, "bic": bic_score, "aic": aic_score, "entropy": entropy_score, "model": model})
              
        # ====================================================
        # SELECT BEST MODEL
        # ====================================================
        #
        # Usually:
        # lower BIC is best
        #
        # ====================================================
        results_df = pd.DataFrame(results)
        best_idx = results_df["bic"].idxmin()
        best_row = results_df.iloc[best_idx]

        return {"results_df": results_df, "best_k": int(best_row["n_archetypes"]), "best_model": best_row["model"]}