import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from _violin import violin



class plot_data_personality_for_pairs:
    def __init__(self, data_dict, model_name, output_dir, fig_height_factor=0.25, font_size=8):
        self.data = data_dict
        self.model_name = model_name
        self.model_data = data_dict[model_name]
        self.output_dir = output_dir
        self.fig_height_factor = fig_height_factor
        self.font_size = font_size

       
        
   
           
    def __call__(self):
        for arch_name, results in self.model_data.items():
            print(f"Plotting {self.model_name} - {arch_name}")

            self.plot_all(arch_name, results)
            self.save_shap_with_metadata(results, arch_name)

          
   # ===============================
    # MASTER PLOT
    # ===============================
    def plot_all(self, arch_name, results):

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"{self.model_name} - {arch_name}", fontsize=12)

        self.plot_confusion_matrix(axs[0], results)
        self.plot_roc(axs[1], results)
        self.plot_fscore(axs[2], results)
        self.PlotRandomPermutation(axs[3], results, class_index=1)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.model_name}_{arch_name}_metrics.pdf")
        plt.close()

       
        self.plot_shap_full(results, arch_name)

    # ===============================
    # MANN WHITNEY
    # ===============================
    def compute_mannwhitney(self,shap_male, shap_female):
        pvalues = []
        for i in range(shap_male.shape[1]):
            stat, p = mannwhitneyu(shap_male[:, i], shap_female[:, i])
            pvalues.append(p)
        return np.array(pvalues)



    #===============================
    # SHAP PER SEX
    #===============================
    def get_shap_per_sex(self, results):

        # --- Concatenate SHAP across folds
        shap_values = np.vstack(results["shap_values"])

        # --- Metadata (same structure as your previous code)
        mice_info = pd.concat(results["mice_information"], axis=0).reset_index(drop=True)

        # --- Indices
        male_idx = mice_info["sex"] == "male"
        female_idx = mice_info["sex"] == "female" 

         # --- Split
        shap_male = shap_values[male_idx.values, :]
        shap_female = shap_values[female_idx.values, :]
        X = results["data_features"]

        X_male = X.iloc[male_idx.values]
        X_female = X.iloc[female_idx.values]

        return shap_male, shap_female, X_male, X_female
    

      # ===============================
    # CONFUSION MATRIX
    # ===============================
    def plot_confusion_matrix(self, ax, results):

        cm = results["confusion_matrix"]
        class_names = ["rest", "archetype"]

        cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_percent,
            annot=True,
            fmt=".2%",
            cmap="Greys",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")   

      # ===============================
    # ROC
    # ===============================
    def plot_roc(self, ax, results):

        fpr = results["FPR"]
        tpr = results["TPR"]
        auc = results["roc_auc_metrics"]

        ax.plot(fpr, tpr, color="black")
        ax.plot([0, 1], [0, 1], "--", color="gray")

        ax.set_title(f"ROC (AUC={auc:.2f})")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)   
   
    # ===============================
    # FSCORE
    # ===============================
    def plot_fscore(self, ax, results):

        fscore = results["fscore"]

        ax.bar(["rest", "archetype"], fscore, color=["gray", "blue"])
        ax.axhline(0.5, linestyle="--", color="black")

        ax.set_title("F-score")
        ax.set_ylim(0, 1)

    # ===============================
    # SHAP IMPORTANCE
    # ===============================
    def plot_shap_importance(self, ax, results):

        shap_values = np.vstack(results["shap_values"])
        X = results["data_features"]

        plt.sca(ax)
        n_features = X.shape[1]
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False,
            max_display=n_features
        )

        ax.set_title("SHAP Importance")

    # ===============================
    # SHAP VIOLIN
    # ===============================
    def plot_shap_violin(self, ax, results):

        shap_values = np.vstack(results["shap_values"])
        X = results["data_features"]

        plt.sca(ax)
        n_features = X.shape[1]
        shap.summary_plot(
            shap_values,
            X,
             plot_type="violin",
            show=False,
            max_display=n_features
        )

        ax.set_title("SHAP Distribution")

    # ===============================
    # SHAP FULL
    # ===============================
    def plot_shap_full(self, results, arch_name):

       shap_values = np.vstack(results["shap_values"])
       X = results["data_features"]
       feature_names = X.columns

       n_features = X.shape[1]
        # ===== SPLIT SEX =====
       shap_male, shap_female, X_male, X_female = self.get_shap_per_sex(results)
       #================Reorder features by importance=================
       importance = np.mean(np.abs(shap_values), axis=0)
       feature_order = np.argsort(importance)[::-1]
       shap_values = shap_values[:, feature_order]
       shap_male = shap_male[:, feature_order]
       shap_female = shap_female[:, feature_order]
       X = X.iloc[:, feature_order]
       X_male = X_male.iloc[:, feature_order]
       X_female = X_female.iloc[:, feature_order]
       feature_names = feature_names[feature_order]

       
       # ===== STATS =====
       pvals = self.compute_mannwhitney(shap_male, shap_female)
       # FDR correction (important!)
       #pvals = multipletests(pvals, method="fdr_bh")[1]

       fig_height = min(12, max(5, n_features * self.fig_height_factor))   # 🔥 key line
       fig_height = 10
       with plt.rc_context({
           "font.size": self.font_size,
           "axes.titlesize": self.font_size,
           "axes.labelsize": self.font_size,
           "xtick.labelsize": self.font_size,
           "ytick.labelsize": self.font_size,
       }):
           fig, axs = plt.subplots(2, 2, figsize=(20, fig_height))
           fig.suptitle(f"{self.model_name} - {arch_name}", fontsize=self.font_size + 2)

           # ===== Importance =====
           plt.sca(axs[0, 0])
           shap.summary_plot(
                  shap_values,
                  X,
                  feature_names=X.columns.tolist(),
                  plot_type="bar",
                  show=False,
                  max_display=n_features
           )
           axs[0, 0].set_title("SHAP Importance")
           axs[0, 0].tick_params(axis="y", labelsize=self.font_size)
           axs[0, 0].tick_params(axis="x", labelsize=3)

           # ===== Violin (no dots) =====
           plt.sca(axs[0, 1])
           shap.summary_plot(
                  shap_values,
                  X,
                  plot_type="violin",
                  show=False,
                  max_display=n_features
           )
           axs[0, 1].set_title("SHAP Distribution")
           axs[0, 1].tick_params(axis="y", labelsize=self.font_size)
           axs[0, 1].tick_params(axis="x", labelsize=3)

            # ======================
            # FEMALE
            # ======================
           plt.sca(axs[1, 0])
           violin(
            shap_female,
            features= X_female,
            feature_names=feature_names,
            max_display=n_features,
            plot_type = "violin",
            sort=False, show=False
            )
           


           axs[1, 0].set_title("Female")
           axs[1, 0].tick_params(axis="y", labelsize=self.font_size)
           axs[1, 0].tick_params(axis="x", labelsize=3)

            # ======================
            # MALE
            # ======================
           plt.sca(axs[1, 1])
           violin(
            shap_male,
            features= X_male,
            feature_names=feature_names,
            max_display=n_features,
            plot_type = "violin",
            sort=False, show=False
            )
           axs[1, 1].set_title("Male")
           axs[1, 1].tick_params(axis="y", labelsize=self.font_size)
           axs[1, 1].tick_params(axis="x", labelsize=3)
           #===============add asterisks for significant features===============
           ax_left = axs[1, 0]
           ax_right = axs[1, 1]

           # get positions of both axes in figure coordinates
           bbox_left = ax_left.get_position()
           bbox_right = ax_right.get_position()

        # x position = middle between the two plots
           x_mid = (bbox_left.x1 + bbox_right.x0) / 2

           for i, p in enumerate(pvals[::-1]):

            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            elif p < 0.1:
                star = "#"
            else:
                continue

            # convert y (data coord) → figure coord
            y_display = ax_left.transData.transform((0, i+0.5))[1]
            y_fig = plt.gcf().transFigure.inverted().transform((0, y_display))[1]

            plt.gcf().text(
        x_mid,
        y_fig,
        star,
        ha="center",
        va="center",
        fontsize=self.font_size
    )

           #==================================================
    

           plt.tight_layout()
           plt.savefig(f"{self.output_dir}/{self.model_name}_{arch_name}_shap.pdf")
           plt.close()
       #===============================
       # SAVE SHAP WITH METADATA
       #===============================
    def save_shap_with_metadata(self, results, arch_name):
        shap_values = np.vstack(results["shap_values"]) 
        X = results["data_features"].reset_index(drop=True)
        mice_info = pd.concat(results["mice_information"], axis=0).reset_index(drop=True)

        # ===== CHECK =====
        assert len(shap_values) == len(mice_info), "Mismatch SHAP vs metadata"
        assert len(X) == len(mice_info), "Mismatch X vs metadata"

        shap_df = pd.DataFrame(shap_values, columns=[f"shap_{col}" for col in X.columns])
        full_df = pd.concat([mice_info,shap_df], axis=1)
        male_df = full_df[mice_info["sex"] == "male"]
        female_df = full_df[mice_info["sex"] == "female"]
        
        #save to excel to each arch
        save_path = os.path.join(self.output_dir, f"{self.model_name}_{arch_name}_shap_values.xlsx")
        full_df.to_excel(save_path, index=False)

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                full_df.to_excel(writer, sheet_name="all_data", index=False)
                male_df.to_excel(writer, sheet_name="males", index=False)
                female_df.to_excel(writer, sheet_name="females", index=False)

        print(f"Saved Excel → {save_path}")
    

 # ===============================
    # RANDOM PERMUTATION TEST
    # ===============================
    def PlotRandomPermutation(self, ax, results, class_index=1):
        plt.sca(ax)
        ax.set_axis_on()
        # ===== actual F-score =====
        fscore_actual = results["fscore"][class_index]

         # ===== shuffled distribution =====
        if class_index == 0:
            shuffle_scores = results["shuffle_fscore_rest"]
            class_name = "rest"
        else:
            shuffle_scores = results["shuffle_fscore_archetype"]
            class_name = "archetype"

        shuffle_scores = np.array(shuffle_scores)

         # ===== p-value =====
        pval = np.mean(shuffle_scores >= fscore_actual)

         # ===== histogram =====
        ax.hist(
            shuffle_scores,
            bins=10,
            alpha=0.7,
            color="blue",
            edgecolor="black"
        )

        # ===== actual score line =====
        ax.axvline(
            fscore_actual,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Actual = {fscore_actual:.2f}"
        )
        
         # ===== text =====
        ax.text(
            fscore_actual,
            ax.get_ylim()[1] * 0.8,
            f"p = {pval:.3f}",
            color="red",
            fontsize=8
        )

        ax.set_title(f"Permutation Test ({class_name})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Shuffled F-score")
        ax.set_ylabel("Frequency")

        ax.tick_params(axis='both', labelsize=7)

        ax.legend(fontsize=7)