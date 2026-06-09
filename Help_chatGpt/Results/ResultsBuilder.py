import pandas as pd
# ============================================================
# RESULTS TABLE
# ============================================================

class ResultsBuilder:

    def build_results_table(
        self,
        mouse_ids,
        labels,
        probabilities,
        X_pca
    ):

        results = pd.DataFrame()

        results = mouse_ids.copy()

        results["Assigned_Archetype"] = labels
        

        for i in range(probabilities.shape[1]):

            results[
                f"Probability_A{i}"
            ] = probabilities[:, i]

        n_pcs = X_pca.shape[1]
        for i in range(n_pcs):
          results[f"PC{i+1}"] = X_pca[:, i]

        return results