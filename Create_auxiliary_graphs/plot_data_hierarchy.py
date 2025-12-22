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

class plot_data_hierarchy:
    def __init__(self, data):
        self.data = data
        self.results = pd.DataFrame()
        self.set_to_plot = pd.DataFrame()
        
   
        
    def compare_two_groups(self,data):  
      results = []        
      dominant_df = data[data["Hierarchy"] == "dominant"]          
      submissive_df = data[data["Hierarchy"] == "submissive"]
      #find the numeric columns
      num_columns = data.select_dtypes(include = [np.number]).columns.tolist()
      #convert into numpy and loop over each hormone
      for col in num_columns:
          x=dominant_df[col].dropna().to_numpy()
          y=submissive_df[col].dropna().to_numpy()
          n1, n2 = x.size, y.size
          #Mann whitney
          u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method = "asymptotic")
          results.append({"hormone":col, "n_dominant": n1, "n_submissive": n2, "U_stat": float(u_stat), "p_value": float(p_val), "mean_dominant": float(np.mean(x)),"mean_submissive": float(np.mean(y)),
                          "sem_dominant": np.std(x, ddof=1) / np.sqrt(len(x)), "sem_submissive": np.std(y, ddof=1) / np.sqrt(len(y))})
      
      #apply fdr 
      
      results_all = pd.DataFrame(results)    
      #add code significance
      results_all = self.add_significance(results_all)
      results_all['p_value_corrected'] = 1
      
      #remove data
      results_sig = results_all[results_all["pv_code"]> -1] # only correction on the intrested data
      results_no_sig = results_all[results_all["pv_code"] == -1] 
      rejected, pvals_corrected, _, _ = multipletests(results_sig["p_value"], alpha=0.05, method="fdr_bh")
      
      results_sig['p_value_corrected'] = pvals_corrected
      
      results_total = pd.concat([results_sig, results_no_sig], axis = 0)
      hormones_to_consider = results_sig['hormone']
      
      return results_total,  hormones_to_consider
  
    def add_significance(self, results):
         conditions = [ results["p_value"] < 0.001, results["p_value"] < 0.01, 
                       results["p_value"] < 0.05, results["p_value"] < 0.1, 
                       results["p_value"] >= 0.1 ]
         choices = [3,2,1,0,-1]
         results["pv_code"] = np.select(conditions,choices,default = 0)
         
         return results
         
    
    
    def plot_bar_plots(self,data,results,hormones,sex):
   
        # pvals = results["p_value"].to_numpy()
        # pvals_corrected = results["p_value_corrected"].to_numpy()
        y = np.arange(len(hormones))
       # alphas = [plot_data.alpha_from_p(p) for p in pvals]
        alphas = [plot_data_hierarchy.significance_from_p(results[results['hormone'] == h]["p_value_corrected"].iloc[0]) for h in hormones]
        
        fig, (ax_left, ax_labels,  ax_right)  = plt.subplots(1, 3, figsize = (8,11), sharey = False, 
                                                gridspec_kw = {"width_ratios": [4, 1.2, 4], "wspace": 0.2}) #was 0.07
        
        for i,h in enumerate(hormones):
            ax_left.barh(y[i], results[results['hormone'] == h]["mean_dominant"], xerr = results[results['hormone'] == h]["sem_dominant"], color ="mediumaquamarine", capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.6)
            ax_right.barh(y[i], results[results['hormone'] == h]["mean_submissive"], xerr = results[results['hormone'] == h]["sem_submissive"], color ="mediumpurple", capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.6)
        
        #max_val = max(male_means.max() + male_sem.max(), female_means.max() + female_sem.max())
        #margin = 2
        for ax in (ax_left, ax_right):
            #ax.set_xscale("symlog")
            ax.grid(False)
            ax.set_ylim(-0.5, len(hormones) - 0.5)

        ax_left.set_xlim(1.5, 0)   # left plot reversed
        ax_right.set_xlim(0, 1.5)
       #for ax, xlabel in zip((ax_left,ax_right),("Male (pg/mg)", "Female (pg/mg)")):
        for ax, xlabel in zip((ax_left,ax_right),("Dominant", "Submissive")):
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
        
        
      
        #add raw data
        self.add_scatter_points(data,ax_left, ax_right, hormones)
        
       
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig("F:/SilviaData/rutiFrishman/September2025/Difference_dom_sub_" +sex + ".pdf", bbox_inches="tight")
        plt.close()
    
    def add_scatter_points(self, data ,ax_left, ax_right, hormones):
        df = data
        y = np.arange(len(hormones))
        # scatter parameters
        rng = np.random.default_rng(42)  # reproducible jitter
        jitter_strength = 0.08
        
        for i, hormone in enumerate(hormones):
          male_vals = df.loc[df["Hierarchy"]=="dominant", hormone]
          female_vals = df.loc[df["Hierarchy"]=="submissive", hormone]
          male_y = np.full(len(male_vals), y[i]) + rng.normal(0, jitter_strength, size=len(male_vals))
          female_y = np.full(len(female_vals), y[i]) + rng.normal(0, jitter_strength, size=len(female_vals))
          ax_left.scatter(male_vals, male_y,
                    facecolors="mediumaquamarine", edgecolors="darkgreen",
                    alpha=0.7, s=7, linewidths=0.8, zorder=2)
    
          ax_right.scatter(female_vals, female_y,
                     facecolors="mediumpurple", edgecolors="indigo",
                     alpha=0.7, s=7, linewidths=0.8, zorder=2)
   
    def normalize_data(self,data):
        df_normalized = data.copy()
        df_normalized.iloc[:,1:] = (data.iloc[:,1:] - data.iloc[:,1:].min()) / (data.iloc[:,1:].max() - data.iloc[:,1:].min())
        return df_normalized
    
    
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
            
    def get_hormones_list(self,h1,h2):
        combined_hormones = pd.concat([h1,h2], axis=0, ignore_index =True)
        unique_hormones = combined_hormones.unique().tolist()
        return unique_hormones
    
    def get_ordered_hormone_list(self,results_male,hormones_list):
        filtered = results_male[results_male['hormone'].isin(hormones_list)]
        results = filtered.sort_values(by = ["pv_code", "mean_dominant"], ascending = [True,True])
        hormone_list =results['hormone']
        return hormone_list
    
    def add_fdr(self,results, hormone_list):
         rejected, pvals_corrected, _, _ = multipletests(results[results["hormone"].isin(hormone_list)]["p_value"], alpha=0.05, method="fdr_bh")
         results[results["hormone"].isin(hormone_list)]['p_value_corrected'] = pvals_corrected
         return results
    
    def __call__(self,hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        data_male = (self.data[self.data["sex"]== "male"]).iloc[:,1:]
        data_female = (self.data[self.data["sex"]== "female"]).iloc[:,1:]
        data_male_normalized = self.normalize_data(data_male)
        data_female_normalized = self.normalize_data(data_female)
        
        #without normalization
        results_male_wn, hormones_maleswn = self.compare_two_groups(data_male)
        results_female_wn, hormones_femaleswn  = self.compare_two_groups(data_female)
        
        results_male, hormones_males = self.compare_two_groups(data_male_normalized)
        results_female, hormones_females  = self.compare_two_groups(data_female_normalized)
        
        #correction
        results_male["p_value"] = results_male_wn["p_value"]
        results_female["p_value"] = results_female_wn["p_value"]
        hormones_males = hormones_maleswn
        hormones_females = hormones_femaleswn
        
        hormones_list = self.get_hormones_list(hormones_males, hormones_females)
        
        #get hormones list according to males
        ordered_hormone_list = self.get_ordered_hormone_list(results_male,hormones_list)
        #get fdr for the ordered_hormone_list
        results_male = self.add_fdr(results_male, ordered_hormone_list)
        results_female = self.add_fdr(results_female, ordered_hormone_list)
        #save data
        with pd.ExcelWriter("F:/SilviaData/rutiFrishman/September2025/dominantvssub.xlsx", engine="openpyxl") as writer:
           results_male.to_excel(writer, sheet_name="male", index=False)
           results_female.to_excel(writer, sheet_name="female", index=False)
        
       #plot males
        sex = "male"
        self.plot_bar_plots(data_male_normalized,results_male,ordered_hormone_list,sex)
        
        sex = "female"
        self.plot_bar_plots(data_female_normalized,results_female,ordered_hormone_list,sex)
       
        a=1
