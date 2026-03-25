import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors

class General_functions:

    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir

    def __call__(self):
        corr_df = self.correlation_matrix()
        self.plot_all_pareto(corr_df)
        corr_df = self.correlation_matrix("female")
        self.plot_all_pareto(corr_df, "female")
        corr_df = self.correlation_matrix("male")
        self.plot_all_pareto(corr_df, "male")


#####################Correlation matrix #########################
    def correlation_matrix(self, sex = None):
        data = pd.read_excel(self.data_path)
        if sex is not None:
            data = data[data['sex'] == sex]

        # Define features and targets based on the structure used in treat_continous_labels
        X = data.drop(
            ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips',
             'Last.day.Glicko','Animal','sexFeature','Arch1','Arch2','Arch3','Arch4'], axis=1)
        y = data[['Arch1','Arch2','Arch3','Arch4']]

        correlations = []
        for target in y.columns:
            for feature in X.columns:
                r, p = spearmanr(X[feature], y[target])
                correlations.append({
                    'Feature': feature,
                    'Arch': target,
                    'Spearman_r': r,
                    'p_value': p
                })

        corr_df = pd.DataFrame(correlations)

        # Add BH correction for p-values
        reject, pvals_corrected, _, _ = multipletests(corr_df['p_value'], alpha=0.1, method='fdr_bh')
        corr_df['p_value_adj'] = pvals_corrected

        if sex is  None:
          corr_df.to_excel(self.output_dir + 'feature_target_correlations.xlsx', index=False)
        else:
           corr_df.to_excel(self.output_dir +  sex  + '_feature_target_correlations.xlsx', index=False) 
        
        return corr_df
    
    ######################Do graph to show significant correlation#################
    def plot_all_pareto(self, corr_df, sex = None):
        archetypes = corr_df['Arch'].unique()
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        if sex is  None:
            fig.suptitle('Significant Spearman Correlations with Archetypes (p < 0.1)', fontsize=16)
        else:
            fig.suptitle(sex + 'Significant Spearman Correlations with Archetypes (p < 0.1)', fontsize=16)
            
        axes = axes.flatten()
        colors = ['pink','purple','orange', 'green']
        for i, arch_name in enumerate(archetypes):
            if i < len(axes):
                self.plot_pareto(corr_df, arch_name, ax=axes[i], color = colors[i])


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if sex is  None:
          plt.savefig(self.output_dir + 'feature_target_correlations_plots.pdf')
        else:
           plt.savefig(self.output_dir +  sex  + '_feature_target_correlations_plots.pdf')

        plt.close()
    
    #######################do the plot for each archetype#########################################
    def plot_pareto(self, corr_df, arch_name, ax, color):
        corr_method = "spearman"

        subset = corr_df[corr_df["Arch"] == arch_name].copy()
        subset = subset[subset['p_value'] < 0.1]
        subset = subset.sort_values("Spearman_r", ascending=True)

        y = np.arange(len(subset))
         # small gap so line doesn't touch the dot
        gap = 0.02
        line_end = subset["Spearman_r"] - np.sign(subset["Spearman_r"]) * gap


        ax.hlines(
            y=y,
            xmin=0,
            xmax=line_end,
            color="black",
            alpha=0.7
        )

        # Set alpha based on p-value: more transparent for 0.05 <= p < 0.1
        alphas = np.where(subset['p_value'] < 0.05, 1.0, 0.4)
        base_color = mcolors.to_rgba(color)
        scatter_colors = [(*base_color[:3], alpha) for alpha in alphas]

        ax.scatter(
            subset["Spearman_r"],
            y,
            color=scatter_colors,
            s=90
        )
  
        ax.set_yticks(y, labels=subset["Feature"])
        ax.axvline(0, color="black")

        ax.set_xlabel(f"{corr_method.capitalize()} correlation")
        ax.set_title(f"{arch_name}")

        # Remove plot borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)

