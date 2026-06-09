# ============================================================
# PERMUTATION TEST
# ============================================================

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict


class PermutationTester:
    def __init__(self, n_iterations=300):
        self.n_iterations = n_iterations

    def run(self,model,X,Y,  true_r2):   
        perm_r2 = np.zeros((self.n_iterations,Y.shape[1]))
        for perm in range(self.n_iterations):
            Y_shuffled = np.copy(Y)
            for i in range(Y.shape[1]):
                np.random.shuffle(Y_shuffled[:, i])
                Y_pred = cross_val_predict(model.model, X, Y_shuffled, cv=LeaveOneOut())
                for i in range(Y.shape[1]):
                   perm_r2[perm, i] = r2_score(Y_shuffled[:, i], Y_pred[:, i])

        p_values = []

        for i in range(Y.shape[1]):
            p = np.mean(perm_r2[:, i] >= true_r2[i])
            p_values.append(p)
        return {"perm_r2": perm_r2, "true_r2": true_r2, "p_values": p_values}
    
    # ========================================================
    # PLOT HISTOGRAMS
    # ========================================================
    def plot_histograms(self, results,archetype_names=None,save_file=None):
        perm_r2 = results["perm_r2"]
        true_r2 = results["true_r2"]
        p_values = results["p_values"]

        n_arch = perm_r2.shape[1]

        if archetype_names is None:

            archetype_names = [f"Archetype_{i}" for i in range(n_arch)]

        fig, axes = plt.subplots(1, n_arch, figsize=(6 * n_arch, 5))

        for i, ax in enumerate(axes):

            # histogram
            ax.hist(perm_r2[:, i], bins=30, alpha=0.8)

            # true model line
            ax.axvline(true_r2[i], linewidth=3, linestyle="--", label=f"True R² = {true_r2[i]:.3f}")
            ax.set_title(archetype_names[i])

            ax.set_xlabel("Permutation R²")

            ax.set_ylabel("Count")

            # text box
            textstr = (
                f"True R² = {true_r2[i]:.3f}\n"
                f"p = {p_values[i]:.5f}"
            )

            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8
                )
            )

            fig.suptitle(
            "Permutation Test",
            fontsize=18
        )

        plt.tight_layout()

        if save_file is not None:
          save_file = f"{save_file}/Permutation_Test_Histograms.tiff"
          plt.savefig(
                save_file,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

