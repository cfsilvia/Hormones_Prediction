import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from statsmodels.stats.multitest import multipletests

class plot_data:
    def __init__(self, data):
        self.data = data
        self.results = pd.DataFrame()
        self.set_to_plot = pd.DataFrame()
        
    def find_mean_sem(self):
        long_data = self.data.melt(id_vars=["sex"], var_name = "Hormone", value_name = "Value")
        agg =(long_data.groupby(['Hormone', "sex"], as_index = False).agg(mean=("Value","mean"), sem = ("Value", plot_data.sem)))
        
        return agg
        
    def compare_two_groups(self):  
      results = []        
      male_df = self.data[self.data["sex"] == "male"]          
      female_df = self.data[self.data["sex"] == "female"]
      #find the numeric columns
      num_columns = self.data.select_dtypes(include = [np.number]).columns.tolist()
      #convert into numpy and loop over each hormone
      for col in num_columns:
          x=male_df[col].dropna().to_numpy()
          y=female_df[col].dropna().to_numpy()
          n1, n2 = x.size, y.size
          #Mann whitney
          u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method = "asymptotic")
          results.append({"hormone":col, "n_male": n1, "n_female": n2, "U_stat": float(u_stat), "p_value": float(p_val), "mean_male": float(np.mean(x)),"mean_female": float(np.mean(y)),
                          "sem_male": np.std(x, ddof=1) / np.sqrt(len(x)), "sem_female": np.std(y, ddof=1) / np.sqrt(len(y))})
      
      #apply fdr 
      
      self.results = pd.DataFrame(results)      
      rejected, pvals_corrected, _, _ = multipletests(self.results["p_value"], alpha=0.05, method="fdr_bh")
      self.results["p_value_corrected"] = pvals_corrected
      a=1
  
    def add_significance(self):
         conditions = [ self.results["p_value"] < 0.001, self.results["p_value"] < 0.01, 
                       self.results["p_value"] < 0.05, self.results["p_value"] < 0.1, 
                       self.results["p_value"] >= 0.1 ]
         choices = [3,2,1,0,-1]
         self.results["pv_code"] = np.select(conditions,choices,default = 0)
         
    def order_results(self,hormones,hormones_ratio,end_cann, end_cann_ratio,aminoacids, aminoacids_ratio):
        set = pd.DataFrame(columns = self.results.columns)
       
        data_names = hormones + hormones_ratio + end_cann + end_cann_ratio + aminoacids + aminoacids_ratio
        #data_names = hormones
        self.ordered_results = self.results.set_index("hormone").loc[data_names].reset_index()
        for data in (aminoacids_ratio, aminoacids, end_cann_ratio, end_cann, hormones_ratio, hormones):
          auxiliary = self.ordered_data(data)
          set = pd.concat([set, auxiliary], ignore_index=True)
        self.set_to_plot = set
       
             
    def ordered_data(self, data):
        auxiliary = self.ordered_results[self.ordered_results["hormone"].isin(data)]
        auxiliary = auxiliary.sort_values(by = ["pv_code", "mean_male"], ascending = [True,True])
        return auxiliary
    
    def plot_bar_plots(self):
        hormones = self.set_to_plot["hormone"].to_numpy()
        male_means = self.set_to_plot["mean_male"].to_numpy()
        female_means = self.set_to_plot["mean_female"].to_numpy()
        male_sem = self.set_to_plot["sem_male"].to_numpy()
        female_sem = self.set_to_plot["sem_female"].to_numpy()
        pvals = self.set_to_plot["p_value"].to_numpy()
        pvals_corrected = self.set_to_plot["p_value_corrected"].to_numpy()
        y = np.arange(len(hormones))
       # alphas = [plot_data.alpha_from_p(p) for p in pvals]
        alphas = [plot_data.significance_from_p(p) for p in pvals_corrected]
        
        fig, (ax_left, ax_labels,  ax_right)  = plt.subplots(1, 3, figsize = (8,11), sharey = False, 
                                                gridspec_kw = {"width_ratios": [4, 1.2, 4], "wspace": 0.2}) #was 0.07
        
        for i in range(len(hormones)):
            ax_left.barh(y[i], male_means[i], xerr = male_sem[i], color ="mediumaquamarine", capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.6)
            ax_right.barh(y[i], female_means[i], xerr = female_sem[i], color ="mediumpurple", capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.6)
        
        max_val = max(male_means.max() + male_sem.max(), female_means.max() + female_sem.max())
        margin = 2
        for ax in (ax_left, ax_right):
            #ax.set_xscale("symlog")
            ax.grid(False)
            ax.set_ylim(-0.5, len(hormones) - 0.5)

        ax_left.set_xlim(1.5, 0)   # left plot reversed
        ax_right.set_xlim(0, 1.5)
       #for ax, xlabel in zip((ax_left,ax_right),("Male (pg/mg)", "Female (pg/mg)")):
        for ax, xlabel in zip((ax_left,ax_right),("Male", "Female")):
            ax.set_xlabel(xlabel)
            ax.xaxis.set_label_position("top")
            ax.xaxis.set_ticks_position("top")
            ax.tick_params(axis="x", which="both", bottom=False)
            ax.spines["top"].set_visible(True)
            ax.spines["bottom"].set_visible(False)
            
        ax_right.set_yticks([])
        ax_right.set_yticklabels([])
        ax_left.set_yticks([])
        ax_left.set_yticklabels([])
        
        ax_left.spines["left"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        
        for ax in (ax_left, ax_right):
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
        
        ax_labels.set_xlim(0,1) 
        ax_labels.set_ylim(-0.5, len(hormones) - 0.5)   
        ax_labels.set_xticks([])
        ax_labels.axis("off")
        
        for i, h in enumerate(hormones):
          ax_labels.text(0.5, i, h + alphas[i],
                   ha="center", va="center",
                   fontsize=8, fontweight="bold")
        for spine in ax_labels.spines.values():
           spine.set_visible(False)
        
        
        # #legends
        # patches = [mpatches.Patch(color="grey", alpha =1.0, label = "p < 0.001"),
        #            mpatches.Patch(color="grey", alpha =0.7, label = "p < 0.01"),
        #            mpatches.Patch(color="grey", alpha =0.5, label = "p < 0.05"),
        #            mpatches.Patch(color="grey", alpha =0.3, label = "p < 0.1"),
        #            mpatches.Patch(color="grey", alpha =0.1, label = "p ≥ 0.1")]
        # fig.legend(handles = patches, loc="lower center",bbox_to_anchor=(0.5, -0.01), ncol = 5, fontsize = 9, frameon = False, title = "p-value")
        
        #add raw data
        self.add_scatter_points(ax_left, ax_right, hormones)
        
        #add separations
       # separators = [9, 14]
        separators = [13, 24]
        for sep in separators:
          for ax in (ax_left, ax_labels, ax_right):
             ax.axhline(sep - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=5)
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig("F:/SilviaData/rutiFrishman/September2025/hormones_plota.pdf", bbox_inches="tight")
        plt.close()
    
    def add_scatter_points(self, ax_left, ax_right, hormones):
        df = self.data
        y = np.arange(len(hormones))
        # scatter parameters
        rng = np.random.default_rng(42)  # reproducible jitter
        jitter_strength = 0.08
        
        for i, hormone in enumerate(hormones):
          male_vals = df.loc[df["sex"]=="male", hormone]
          female_vals = df.loc[df["sex"]=="female", hormone]
          male_y = np.full(len(male_vals), y[i]) + rng.normal(0, jitter_strength, size=len(male_vals))
          female_y = np.full(len(female_vals), y[i]) + rng.normal(0, jitter_strength, size=len(female_vals))
          ax_left.scatter(male_vals, male_y,
                    facecolors="mediumaquamarine", edgecolors="darkgreen",
                    alpha=0.7, s=7, linewidths=0.8, zorder=2)
    
          ax_right.scatter(female_vals, female_y,
                     facecolors="mediumpurple", edgecolors="indigo",
                     alpha=0.7, s=7, linewidths=0.8, zorder=2)
    #-------------------------------------------------
    def plot_bar_plots_broken_axis(self):
        # # hormones ration
        LOW = (0,0.2)
        HIGH = (1.5,10)
        
        # # endocann ration
        # LOW = (0,90)
        # HIGH = (800,3000)
        
         # # amino ration
        # LOW = (0,1.5)
        # HIGH = (1.8,100)
        
        #hormones
        # LOW  = (0, 15) 
        # HIGH = (24, 50)  
        #aminoacids
        #LOW  = (0, 20000) 
        #HIGH = (25000, 50000)  
        #endoc
        # LOW  = (0, 3000) 
        # HIGH = (20000, 120000) 
        hormones = self.set_to_plot["hormone"].to_numpy()
        male_means = self.set_to_plot["mean_male"].to_numpy()
        female_means = self.set_to_plot["mean_female"].to_numpy()
        male_sem = self.set_to_plot["sem_male"].to_numpy()
        female_sem = self.set_to_plot["sem_female"].to_numpy()
        pvals = self.set_to_plot["p_value"].to_numpy()
        
        y = np.arange(len(hormones))
        alphas = [plot_data.significance_from_p(p) for p in pvals]
        
        fig = plt.figure(figsize=(8,4))
        gs = GridSpec(1, 3,width_ratios=[4, 1, 4], wspace=0.05)
        #hormones (50,20),(15,0) (0,15),(20,50)
        ax_left = brokenaxes(xlims=((HIGH[1], HIGH[0]), (LOW[1], LOW[0])),wspace=0.2, subplot_spec=gs[0, 0])
        ax_labels = fig.add_subplot(gs[0, 1]); ax_labels.axis("off")
        ax_right = brokenaxes(xlims=((LOW[0], LOW[1]), (HIGH[0], HIGH[1])), wspace=0.2,subplot_spec=gs[0, 2])
        
        for i in range(len(hormones)):
            ax_left.barh(y[i], male_means[i], xerr = male_sem[i], color ="mediumaquamarine", capsize =4, 
                         error_kw={"elinewidth":1, "capthick":1,"zorder": 3}, height = 0.5)
            ax_right.barh(y[i], female_means[i], xerr = female_sem[i], color ="mediumpurple", capsize =4, 
                          error_kw={"elinewidth":1, "capthick":1,"zorder": 3}, height = 0.5)
            
        # set y-limits once on the BrokenAxes containers
        ax_left.set_ylim(-0.5, len(hormones) - 0.5)
        ax_right.set_ylim(-0.5, len(hormones) - 0.5)
        for a in ax_left.axs:
          a.spines["left"].set_visible(False)
          a.xaxis.tick_top()
          a.xaxis.set_label_position("top")
          a.tick_params(axis="x",top=True, labeltop=True,bottom=False, labelbottom=False, labelsize=5, pad =2)
          a.set_yticks([])
          a.set_yticklabels([])
          
        for a in ax_right.axs:
            a.xaxis.tick_top()
            a.xaxis.set_label_position("top")
           
            a.tick_params(axis="x",top=True, labeltop=True,bottom=False, labelbottom=False, labelsize=5, pad =2)
          
        ax_left.axvline(x=0, color="black", linewidth = 1)
        
       
        # add x-axis labels (on the BrokenAxes containers) and nudge them upward a bit
        ax_left.set_xlabel("Male")
        ax_right.set_xlabel("Female")
        
        
        # center labels column
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(-0.5, len(hormones) - 0.5)
        ax_labels.set_xticks([])
        
        
      
        
        for i, h in enumerate(hormones):
          ax_labels.text(0.5, i, h + alphas[i],
                   ha="center", va="center",
                   fontsize=5, fontweight="bold")
        for spine in ax_labels.spines.values():
           spine.set_visible(False)
        
        
        # # #legends
        # patches = [mpatches.Patch(color="grey", alpha =1.0, label = "p < 0.001"),
        #            mpatches.Patch(color="grey", alpha =0.7, label = "p < 0.01"),
        #            mpatches.Patch(color="grey", alpha =0.5, label = "p < 0.05"),
        #            mpatches.Patch(color="grey", alpha =0.3, label = "p < 0.1"),
        #            mpatches.Patch(color="grey", alpha =0.1, label = "p ≥ 0.1")]
        # fig.legend(handles = patches, loc="lower center",bbox_to_anchor=(0.5, -0.01), ncol = 5, fontsize = 9, frameon = False, title = "p-value")
        
        #add raw data
        self.add_scatter_points(ax_left, ax_right, hormones)
       
        for a in ax_left.axs:
           a.set_yticks([])
           a.set_yticklabels([])
        for a in ax_right.axs:
           a.set_yticks([])
           a.set_yticklabels([])
        
        plt.tight_layout(rect=[0, 0.12, 1, 0.94])
        plt.savefig("F:/SilviaData/rutiFrishman/September2025/hormones_plot_b.pdf", bbox_inches="tight")
        plt.close()
    #-------------------------------------------
    def normalize_data(self):
        df_normalized = self.data.copy()
        df_normalized.iloc[:,1:] = (self.data.iloc[:,1:] - self.data.iloc[:,1:].min()) / (self.data.iloc[:,1:].max() - self.data.iloc[:,1:].min())
        self.data =df_normalized
    
    
    @staticmethod
    def sem(x):
        x= pd.Series(x)
        return x.std(ddof=1)/np.sqrt(len(x))
    
    @staticmethod
    def alpha_from_p(p):
        if p < 0.001: 
            alpha = 1.0
        elif p < 0.01:
            alpha = 0.7
        elif p < 0.05:
            alpha = 0.5
        elif p < 0.1:
            alpha = 0.3
        else:
            alpha = 0.1
        return alpha
            
    @staticmethod
    def significance_from_p(p):
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
        self.normalize_data()
        #aggregate the data 
        agg = self.find_mean_sem()
        self.compare_two_groups()
        self.add_significance()
        self.order_results(hormones,hormones_ratio,end_cann, end_cann_ratio,aminoacids, aminoacids_ratio)
        #self.order_results(hormones, end_cann, aminoacids)
        self.set_to_plot.to_excel("F:/SilviaData/rutiFrishman/September2025/list_all.xlsx", index =False)
        self.plot_bar_plots()
       # self.plot_bar_plots_broken_axis()
        a=1
