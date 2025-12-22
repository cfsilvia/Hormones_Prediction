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
from statsmodels.stats.multitest import fdrcorrection_twostage


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
      rejected, pvals_corrected, _, _ = multipletests(results_sig["p_value"] , alpha=0.1, method="fdr_bh")
      #rejected, pvals_corrected, _, _ = fdrcorrection_twostage(results_sig["p_value"], alpha=0.05, method='bh')
      
      results_sig['p_value_corrected'] = pvals_corrected
      
      results_total = pd.concat([results_sig, results_no_sig], axis = 0)
      #results_total['p_value_corrected'] = pvals_corrected
      hormones_to_consider = results_sig['hormone']
     # hormones_to_consider = results_total['hormone']
      
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
        y = np.arange(len(hormones))*0.5
       # alphas = [plot_data.alpha_from_p(p) for p in pvals]
        alphas = [plot_data_hierarchy.significance_from_p(results[results['hormone'] == h]["p_value"].iloc[0]) for h in hormones]
        
        fig, ax  = plt.subplots(figsize = (4,6))
        if sex == "male":
            color = "mediumaquamarine"
        else:
            color = "mediumpurple"
            
        for i,h in enumerate(hormones.to_list()):
            ax.barh(y[i], -results[results['hormone'] == h]["mean_dominant"], xerr = results[results['hormone'] == h]["sem_dominant"], color = color, capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.3 )
            ax.barh(y[i], results[results['hormone'] == h]["mean_submissive"], xerr = results[results['hormone'] == h]["sem_submissive"], color = color, capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},height = 0.3,  alpha=0.4 )
        
        #max_val = max(male_means.max() + male_sem.max(), female_means.max() + female_sem.max())
        #margin = 2
        ax.grid(False)
        ax.set_ylim(-0.5, len(hormones)*0.5 )

        for i, h in enumerate(hormones.to_list()):
           ax.text(-1*1.15, y[i], h + alphas[i],
                   ha="right", va="center",
                   fontsize=8, fontweight="bold")
      
        # make both sides display positive x labels
        xticks = np.linspace(0, 1.0, 5)[1:]  # choose tick positions (e.g., 0.5, 1.0, 1.5)
        ax.set_xticks(list(-xticks) + list(xticks))
        ax.set_xticklabels([f"{abs(x):.2g}" for x in list(-xticks) + list(xticks)], fontsize=8)
        ax.set_yticks([])
        #ax.set_xlabel("Concentration (pg/mg)")
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # headers over halves
        ax.text(0.25, 1.06, "Dominant",   transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(0.75, 1.06, "Submissive", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
        
      
        #add raw data
        self.add_scatter_points(data,ax, hormones,sex)
        
        plt.axvline(0, color='black',linewidth=1)  # vertical line at 0 (like in your PDF)
        plt.xlim(-1.03, 1.03)              
        ax.invert_yaxis()
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/Difference_dom_sub_" +sex + ".pdf", bbox_inches="tight")
        plt.close()
    
    def add_scatter_points(self,data, ax, hormones,sex):
        df = data
        y = np.arange(len(hormones))*0.5
        # scatter parameters
        rng = np.random.default_rng(42)  # reproducible jitter
        jitter_strength = 0.08
        
        
        if sex == "male":
            color = "mediumaquamarine"
            edgecolor = "darkgreen"
        else:
            color = "mediumpurple"
            edgecolor = "indigo"
        
        for i, hormone in enumerate(hormones):
          male_vals = df.loc[df["Hierarchy"]=="dominant", hormone]
          female_vals = df.loc[df["Hierarchy"]=="submissive", hormone]
          male_y = np.full(len(male_vals), y[i]) + rng.normal(0, jitter_strength, size=len(male_vals))
          female_y = np.full(len(female_vals), y[i]) + rng.normal(0, jitter_strength, size=len(female_vals))
          ax.scatter(-male_vals, male_y,
                    facecolors=color, edgecolors=edgecolor,
                    alpha=0.7, s=7, linewidths=0.8, zorder=2)
    
          ax.scatter(female_vals, female_y,
                     facecolors=color, edgecolors=edgecolor,
                     alpha=0.6, s=7, linewidths=0.8, zorder=2)
   
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
        
        # #correction
        # results_male["p_value"] = results_male_wn["p_value"]
        # results_female["p_value"] = results_female_wn["p_value"]
        # hormones_males = hormones_maleswn
        # hormones_females = hormones_femaleswn
        
        # hormones_list = self.get_hormones_list(hormones_males, hormones_females)
        
        # #get hormones list according to males
        # ordered_hormone_list = self.get_ordered_hormone_list(results_male,hormones_list)
        # #get fdr for the ordered_hormone_list
        # results_male = self.add_fdr(results_male, ordered_hormone_list)
        # results_female = self.add_fdr(results_female, ordered_hormone_list)
        #save data
        with pd.ExcelWriter("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/dominantvssub.xlsx", engine="openpyxl") as writer:
           results_male.to_excel(writer, sheet_name="male", index=False)
           results_female.to_excel(writer, sheet_name="female", index=False)
        
       #plot males
        sex = "male"
       
        self.plot_bar_plots(data_male_normalized,results_male,hormones_males,sex)
        
        sex = "female"
        
        self.plot_bar_plots(data_female_normalized,results_female,hormones_females,sex)
       
        a=1
