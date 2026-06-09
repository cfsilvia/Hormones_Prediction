import os

from matplotlib import cm, pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

class ArchetypeCorrelationAnalyzer:

    def compute_correlations(
        self,
        behavior_df,
        probabilities
    ):

        rows = []

        n_arch = probabilities.shape[1]

        for arch in range(n_arch):

            prob = probabilities[:, arch]
            arch_rows = []

            for feature in behavior_df.columns:

                r, p = pearsonr(
                    prob,
                    behavior_df[feature]
                )

                arch_rows.append({

                    "Archetype": arch,

                    "Feature": feature,

                    "Correlation_r": r,

                    "P_value": p
                })
            arch_df = pd.DataFrame(arch_rows)
            #BH / FDR correction do for each archetype the correction
            rejected, corrected_pvals, _, _ = multipletests( arch_df["P_value"],alpha = 0.05, method="fdr_bh")
            arch_df["BH_FDR_Pvalue"] = corrected_pvals
            arch_df["Significant_FDR"] = rejected
            rows.append(arch_df)

        correlation_df = pd.concat(rows, ignore_index=True)

        return correlation_df
    
    def top_features(
        self,
        correlation_df,
       alpha = 0.05
    ):

        results = []

        for arch in correlation_df["Archetype"].unique():

            sub = correlation_df[correlation_df["Archetype"] == arch].copy()
            #keep only significant correlations
            sub = sub[sub["BH_FDR_Pvalue"] < alpha]
             # Sort positive correlations
            positive = sub[sub["Correlation_r"] > 0].sort_values("Correlation_r",ascending=False)

            # Sort negative correlations
            negative = sub[sub["Correlation_r"] < 0].sort_values("Correlation_r", ascending=True)

            combined = pd.concat([positive, negative])
            combined["Archetype"] = arch
            results.append(combined)
        results_df = pd.concat(results, ignore_index=True)

        return results_df

    def plot_significant_correlations(self,correlation_df, output_dir = None):

        # Keep only significant correlations
        sig_df = correlation_df[correlation_df["BH_FDR_Pvalue"] < 0.05].copy()

        # Get archetypes
        archetypes = sorted(sig_df["Archetype"].unique())

        # Create subplot per archetype
        fig, axes = plt.subplots(1,len(archetypes),figsize=(40,20), constrained_layout=True)
        # Colormap for archetypes
        cmap = cm.get_cmap('viridis', len(archetypes))
         
        cluster_colors = [
                    "#440154",  # purple
                    "#1b7837",  # green
                    "#d95f02" ,  # orange instead of yellow
                    "#31688e",  # blue
                    ]

        cmap = ListedColormap(cluster_colors[: len(archetypes)])


        for ax, arch in zip(axes, archetypes):

            # Filter archetype
            sub = sig_df[sig_df["Archetype"] == arch].copy()

            # Sort by correlation value
            sub = sub.sort_values(by="Correlation_r",ascending=True)

            # Colors for positive/negative
            colors = ["red" if r > 0 else "blue" for r in sub["Correlation_r"]]

            # Vertical bar plot
            #ax.barh(x=sub["Correlation_r"],y=sub["Feature"],color=colors)
            # horizontal lines
            ax.hlines(y=sub["Feature"],xmin=0, xmax=sub["Correlation_r"], color=colors,linewidth=2)

            # balls at the end
            ax.scatter(sub["Correlation_r"], sub["Feature"],color=colors, s=80,zorder=3,  rasterized=False )

            # Zero line
            ax.axvline(
                0,
                color="black",
                linewidth=1
            )

            ax.set_xlim(-1, 1)


            # Labels
            ax.set_title(
                f"Archetype {arch}",
                fontsize=16,
                fontweight="bold",
                 color=cmap(arch)
            )

            ax.set_xlabel(
                "Pearson r",
                fontsize=10
            )

            # Rotate feature names
            ax.set_yticklabels(
                sub["Feature"],
                rotation=0, 
                fontsize=10,
                fontweight="bold"
            )
            ax.tick_params(
                axis='x',
                labelsize=10,
                rotation=90
            )

            ax.tick_params(
                    axis='y',
                    labelsize=14
                )



        # Global title
        plt.suptitle(
            "Significant Correlations (BH-FDR < 0.05)",
            fontsize=16,
            fontweight="bold",
             y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        pdf_filename = os.path.join(output_dir, f"Correlations_{len(archetypes)}clusters.pdf")
        fig.savefig(pdf_filename, format="pdf",  dpi=300 ,bbox_inches="tight")

        
        plt.show()