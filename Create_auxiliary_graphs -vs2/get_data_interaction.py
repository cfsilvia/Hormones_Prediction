import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,          # important if you ever turned it on
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests


class get_data_interaction:
     def __init__(self, data,sex):
        self.data = data
        self.corr_reordered = pd.DataFrame()
        self.sex = sex
        
     '''
     calculate correlation and pvalue
     '''
     def _pairwise_corr_pvals(self, df):
        cols = df.columns
        n = len(cols)
        corr = pd.DataFrame(np.eye(n), index=cols, columns=cols, dtype=float)
        pval = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols, dtype=float)

        for i in range(n):
            for j in range(i+1, n):
                # pairwise non-null for the two columns
                sub = df[[cols[i], cols[j]]].dropna()
                if len(sub) >= 3:  # need at least 3 points for Pearson r
                    r, p = pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
                else:
                    r, p = np.nan, np.nan
                corr.iat[i, j] = corr.iat[j, i] = r
                pval.iat[i, j] = pval.iat[j, i] = p
            pvals = pval.values.flatten()
            rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.1, method="fdr_bh")
            pvals_corrected_df = pd.DataFrame(pvals_corrected.reshape(pval.shape), index=pval.index, columns=pval.columns)
            
        return corr, pvals_corrected_df
     
     '''
     the idea is to calculate the correlation between all compounds
     and then to cluster together data which has similar values of correlation
     '''   
     def get_correlation(self):
         auxiliary = self.data.iloc[:,1:]
         corr, pval = self._pairwise_corr_pvals(auxiliary) #this is pearson correlation
         #get clustered order using clustermap
         clustergrid = sns.clustermap(corr, cmap="coolwarm", center=0, cbar_pos=None)
         row_order = clustergrid.dendrogram_row.reordered_ind
         col_order = clustergrid.dendrogram_col.reordered_ind
         plt.close(clustergrid.fig)  # close the clustermap figure
         #reorder correlation matrix
         # reorder both correlation and p-value matrices
         ordered_rows = corr.index[row_order]
         ordered_cols = corr.columns[col_order]
         self.corr_reordered = corr.loc[ordered_rows, ordered_cols]
         self.pval_reordered = pval.loc[ordered_rows, ordered_cols]
         
     def _p_to_stars(self, p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "#"
        else:
            return ""
         
     def plot_heatmap(self):
        

         # mask lower triangle (keep diagonal visible, optional)
         mask = np.tril(np.ones_like(self.corr_reordered, dtype=bool), k=-1)
         plt.figure(figsize=(10,8))
         ax = sns.heatmap(self.corr_reordered,mask = mask, annot=False, 
                          cmap="coolwarm", center=0, linewidths=0.5,  vmin=-1, vmax=1,
                          square=True,cbar_kws=dict(shrink=0.8, label="Pearson correlation"))
         plt.title(self.sex)
         plt.xticks(fontsize=10, rotation=60)  # x-axis labels
         plt.yticks(fontsize=10, rotation=0)   # 
         
         # Adjust colorbar font sizes
         cbar = ax.collections[0].colorbar
         cbar.ax.tick_params(labelsize=6)  # smaller tick labels
         cbar.set_label("Pearson correlation", fontsize=8)  # smaller colorbar label
         
         nrows, ncols = self.corr_reordered.shape
         for i in range(nrows):
            for j in range(i+1, ncols):  # j > i => upper triangle
                stars = self._p_to_stars(self.pval_reordered.iat[i, j])
                if stars:
                    ax.text(
                        j + 0.5, i + 0.5, stars,
                        ha="center", va="center", fontsize=11, fontweight="bold"
                    )

        # make a small legend for asterisks
         legend_elements = [
             Line2D([0], [0], marker='', linestyle='None', markersize=8, label='*** p < 0.001'),
             Line2D([0], [0], marker='', linestyle='None', markersize=8, label='** p < 0.01'),
             Line2D([0], [0], marker='', linestyle='None', markersize=8, label='*  p < 0.05'),
             Line2D([0], [0], marker='', linestyle='None', markersize=8, label='#  p < 0.1'),  
           
        ]
         
         # --- Color tick labels by range ---
        # X-axis
         for i, label in enumerate(ax.get_xticklabels()):
                if i < 4:
                    label.set_color("red")
                elif i < 9:
                    label.set_color("green")
                else:
                    label.set_color("purple")

            # Y-axis
         for i, label in enumerate(ax.get_yticklabels()):
                if i < 4:
                    label.set_color("red")
                elif i < 9:
                    label.set_color("green")
                else:
                    label.set_color("purple")

         
         
         
         ax.legend(handles=legend_elements, title="", loc='upper left', bbox_to_anchor=(1.02, 0.05), borderaxespad=0.)
         plt.tight_layout()
         plt.savefig(f"U:/Users/Silvia/RutiFrishman_2025_hormones_paper/{self.sex}_correlation_heatmap_font.pdf", format="pdf", bbox_inches="tight")
         plt.show()
    
        
     def __call__(self):
        # self.get_correlation()
         auxiliary = self.data.iloc[:,1:]
         corr, pval = self._pairwise_corr_pvals(auxiliary) 
         self.corr_reordered = corr
         self.pval_reordered = pval

         output = f"U:/Users/Silvia/RutiFrishman_2025_hormones_paper/{self.sex}_correlation_heatmap.xlsx"
            
         
         with pd.ExcelWriter(output, engine="openpyxl") as writer:
            self.corr_reordered.to_excel(writer, sheet_name="correlation", index=False)
            self.pval_reordered.to_excel(writer, sheet_name="pvalue_correction", index=False)

         self.plot_heatmap()
       