import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

class VisualizationTools:

    @staticmethod
    def plot_confusion_matrix(cm, class_names, output_dir):
        cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        cm_percent *= 100
        labels = np.array([[f"{cm_percent[i, j]:.1f}%\n(n={cm[i, j]})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm_percent, annot=labels, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names, linewidths=1, linecolor="white", square=True) 
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.pdf", dpi =300, format = "pdf", bbox_inches="tight")
        plt.close()
        
    @staticmethod
    def plot_permutation(permutation_scores, score, pvalue, output_dir):
        plt.figure(figsize=(7, 5))

        plt.hist(
        permutation_scores,
        bins=30
         )

        plt.axvline(
        score,
        color="red",
        linewidth=3
       )

        plt.xlabel("Permutation Scores")

        plt.ylabel("Count")

        plt.title( f"Permutation Test\n"
    f"Macro F1 = {float(score):.3f} | "
    f"p-value = {float(pvalue):.5f}",
    fontsize=12,
    weight="bold"
)

        plt.tight_layout()   
        plt.savefig(f"{output_dir}/permutation_test.pdf", dpi =300, format = "pdf", bbox_inches="tight")
        plt.close()
    

    @staticmethod
    @staticmethod
    def plot_permutation_per_class(results_class, output_dir):
         n_classes = len(results_class)
         fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 5))
         for ax, (class_name, res) in zip(axes, results_class.items()):
             score = res["score"]
             pvalue = res["pvalue"]
             permutation_scores = res["permutation_scores"]
             ax.hist(permutation_scores, bins=30)
             ax.set_xlabel("Permutation Scores")
             ax.set_ylabel("Count")
             ax.set_title(f"Arch{class_name}\n"
            f"F1 = {float(score):.3f} | "
            f"p = {float(pvalue):.5f}",
            fontsize=12,
            weight="bold")
             ax.axvline(score, color="red", linewidth=3)
         plt.tight_layout()
         plt.savefig(f"{output_dir}/permutation_test_per_class.pdf", dpi=300, format="pdf", bbox_inches="tight")

         plt.close()
             
