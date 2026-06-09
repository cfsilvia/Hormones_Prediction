from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# HORMONE PREDICTION MODEL
# ============================================================

from sklearn.cross_decomposition import PLSRegression


class HormonePredictionModel:
     def __init__(self):
         self.model = PLSRegression(n_components=3) 
    
     def cross_validated_prediction(self, X, Y):
        Y_pred = cross_val_predict(self.model, X, Y,cv=LeaveOneOut())
        return Y_pred
  

     def compute_r2(self, Y_true, Y_pred):
         r2_scores = []
         for i in range(Y_true.shape[1]):
             r2 = r2_score(Y_true[:, i],Y_pred[:, i])
             r2_scores.append(r2)
         return r2_scores
     
     def plot_predictions(self, Y_true, Y_pred, archetype_names=None, save_path=None):
          n_arch = Y_true.shape[1]
          if archetype_names is None:
                archetype_names = [f"Archetype {i}" for i in range(n_arch)]

          fig, axes = plt.subplots(1, n_arch, figsize=(6 * n_arch, 6))

          for i, ax in enumerate(axes):
                 r, p = pearsonr(Y_true[:, i],Y_pred[:, i])
                 r2 = r2_score(Y_true[:, i], Y_pred[:, i])
                 ax.scatter(Y_true[:, i],Y_pred[:, i], s=70)

                # identity line
                 minv = min(Y_true[:, i].min(),Y_pred[:, i].min())
                 maxv = max(Y_true[:, i].max(),Y_pred[:, i].max())
                 ax.plot([minv, maxv], [minv, maxv],linestyle="--", color="red")

                 ax.set_title(archetype_names[i]+ f"\nR={r:.2f}, R2={r2:.2f}, p={p:.3f}")
                 ax.set_xlabel("True Probability")
                 ax.set_ylabel("Predicted Probability")

          plt.tight_layout()
          
          if save_path is not None:
                 save_file = f"{save_path}/Hormone_Prediction.tiff"
                 plt.savefig(save_file, dpi=300,bbox_inches="tight")
          
          plt.show()

    # ============================================================
    # BASELINE MODEL
    # ============================================================
    
     def compute_baseline_r2(self, Y):
             baseline_r2 = []
             for i in range(Y.shape[1]):
                  mean_prob = np.mean(Y[:, i])
                  pred = np.ones(len(Y)) * mean_prob
                  r2 = r2_score(Y[:, i], pred)
                  baseline_r2.append(r2)
             return baseline_r2