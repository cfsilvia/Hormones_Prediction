import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,          # important if you ever turned it on
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
import matplotlib.pyplot as plt 
from scipy import stats
from matplotlib.colors import to_rgba
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

class plot_glycko_correlation:
    def __init__(self, data):
        self.data = data
        self.results_to_plot = pd.DataFrame()
        
    '''
    find correlation between coumpound and last day glyco
    '''    
    def find_correlation(self, sex, list_compounds):
        columns = ["sex", "Compounds", "correlation", "pvalue"]
        df = pd.DataFrame(columns=columns)
        results = []
        aux = self.data[self.data['sex'] == sex]
        
        for i, h in enumerate(list_compounds):
            x = aux[h]
            y = aux["Last.day.Glicko"]
            mask = ~np.isnan(x) & ~np.isnan(y)
            r, p_value = stats.pearsonr(x[mask], y[mask])
            results.append([sex,h, r, p_value])
        df = pd.DataFrame(results, columns=columns).reset_index(drop=True)
        #adjust pvalue
        reject, p_adj, _, _ = multipletests(df["pvalue"], alpha=0.05, method='fdr_bh')
        df["pvalue_adj"] = p_adj
        
        return df
            
    def plot_data(self,compounds):
        size_min=30
        size_max=400    
        df = self.results_to_plot.copy()
        col_order = ["male", "female"]
        df["Compounds"] = pd.Categorical(df["Compounds"], categories = compounds)
        df["sex"] = pd.Categorical(df["sex"], categories = col_order  )
        
        # sizes
        max_abs = df['correlation'].abs().max()
        sizes = size_min + (size_max - size_min) * (df['correlation'].abs() / max_abs)
        df['size'] = sizes
        #colors
        base_colors = np.where(df['correlation'] >= 0, 'tab:red', 'tab:blue')
        # alpha from p-value (smaller p → more opaque)
        alpha = 1 - np.sqrt(df['pvalue'].values)
        alpha = np.clip(alpha, 0.2, 1.0)
        df['rgba'] = [to_rgba(c, a) for c, a in zip(base_colors, alpha)]
        #x and y
        row_order = compounds
        row_index = {r:i for i, r in enumerate(row_order)}
        col_index = {c:i for i, c in enumerate(col_order)}
        x = df['sex'].map(col_index).astype(float).values
        y = df["Compounds"].map(row_index).astype(float).values
        
        # --- 4) Plot
        fig_h = max(4, 0.35 * len(row_order))
        fig_w = max(4, 1.2 * len(col_order))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        
        ax.scatter(x, y, s=df['size'].values, c=list(df['rgba']), edgecolor='k', linewidths=0.5)
        
        for (xi, yi, p, sz) in zip(x, y, df['pvalue'].values, df['size'].values):
          stars = plot_glycko_correlation.p_to_stars(p)  
          if stars:
              fs = max(7, min(14, 6 + 0.02 * np.sqrt(sz)))
              ax.text(xi, yi, stars, ha='center', va='center',
              color='black', fontweight='bold', fontsize=fs, clip_on=True)
              
  
         # Axis ticks & labels
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order)
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_xlim(-0.5, len(col_order)-0.5)
        ax.set_ylim(-0.5, len(row_order)-0.5)
        
        ax.set_title('Correlation with glycko last day heatmap')
        ax.grid(axis='x', linestyle=':', alpha=0.3)
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        ax.invert_yaxis()
        
        ##################
        # --- Legends: color (sign) ---
        sign_handles = [
            Patch(facecolor='tab:red', edgecolor='k', label='Positive corr'),
            Patch(facecolor='tab:blue', edgecolor='k', label='Negative corr')
        ]
        leg_sign = ax.legend(handles=sign_handles, title='', loc='lower left', bbox_to_anchor=(1.05, 0.8), prop={'size': 8},  frameon=True)

        # --- Legend: size (|correlation|) ---
        # choose a few representative |r| values (adjust if you prefer)
        # rep_abs_r = np.array([0.2, 0.5, 0.8]) * max_abs if max_abs > 0 else np.array([0.0])
        # rep_sizes = size_min + (size_max - size_min) * (rep_abs_r / max_abs if max_abs > 0 else 0)

        # size_handles = [ax.scatter([], [], s=s, edgecolor='k', facecolor='none') for s in rep_sizes]
        # size_labels = [f'|r|={v/max_abs:.1f}' if max_abs > 0 else '|r|=0.0' for v in rep_abs_r]

        # leg_size = ax.legend(size_handles, size_labels, title='Effect size', loc='lower right', scatterpoints=1, frameon=True)
        # ax.add_artist(leg_sign)  # keep both legends

        # --- Colorbar: p-value (opacity) ---
        # Your plot uses alpha = 1 - sqrt(p). We can show a p-value scale; lower p = darker bar.
        norm = Normalize(vmin=0.0, vmax=1.0)  # p-value in [0,1]
        sm = ScalarMappable(norm=norm, cmap='Greys_r')
        sm.set_array([])  # required for older Matplotlibs
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.5, aspect=40)
        cbar.set_label('p-value ')
        ##########################       
        
        
        
        
        
        
        
        plt.tight_layout()
        plt.savefig("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/correl_glycko_vs_compound_fonts.pdf", bbox_inches="tight")
        plt.close()


    @staticmethod
    def p_to_stars(p):
        if p < 0.001: 
            alpha = "***"
        elif p < 0.01:
            alpha = "**"
        elif p < 0.05:
            alpha = "*"
        elif p < 0.1:
            alpha = "#"
        else:
            alpha = ""
        return alpha    
        
        
            
    def __call__(self,hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        #finde the correlation for each compound and sex get a frame a row for each compound with rsquare pvalue and pvalue adjust
        #then combine two tables
        list_compounds = hormones + hormones_ratio + end_cann + end_cann_ratio + aminoacids + aminoacids_ratio
        sex ="male"
        df_males = self.find_correlation(sex, list_compounds)
        sex ="female"
        df_females = self.find_correlation(sex, list_compounds)
        
        self.results_to_plot = pd.concat([df_males, df_females], ignore_index = True)
        self.plot_data(list_compounds)
        with pd.ExcelWriter("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/correl_glycko_vs_compound.xlsx", engine="openpyxl") as writer:
           self.results_to_plot.to_excel(writer, sheet_name="data", index=False)
        
        a=1
        
        