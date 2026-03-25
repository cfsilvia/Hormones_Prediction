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
from scipy.stats import mannwhitneyu
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection_twostage
from matplotlib.patches import Patch

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
      results_sig = results_all[results_all["pv_code"]> 0] # only correction on the intrested data
      results_no_sig = results_all[results_all["pv_code"] <= 0] 
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
         
    
    
    def plot_bar_plots(self,data,results,hormones,sex,ax):
        bar_width = 0.4
        # pvals = results["p_value"].to_numpy()
        # pvals_corrected = results["p_value_corrected"].to_numpy()
        x = np.arange(len(hormones))
       # alphas = [plot_data.alpha_from_p(p) for p in pvals]
        alphas = [plot_data_hierarchy.significance_from_p(results[results['hormone'] == h]["p_value"].iloc[0]) for h in hormones]
        
       # fig, ax  = plt.subplots(figsize = (4,6))
        if sex == "male":
            color = "mediumaquamarine"
        else:
            color = "mediumpurple"
            
        for i,h in enumerate(hormones):
            ax.bar(x[i] - bar_width/2, results[results['hormone'] == h]["mean_dominant"], yerr = results[results['hormone'] == h]["sem_dominant"], width = bar_width, color = color, capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3} )
            ax.bar(x[i] + bar_width/2, results[results['hormone'] == h]["mean_submissive"], yerr = results[results['hormone'] == h]["sem_submissive"], width = bar_width, color = color, capsize =4, error_kw={"elinewidth":1, "capthick":1,"zorder": 3},  alpha=0.4 )
        
        #max_val = max(male_means.max() + male_sem.max(), female_means.max() + female_sem.max())
        #margin = 2
        ax.grid(False)
        ax.set_ylim(0,1.5)
        yticks = [0.0, 0.5, 1.0, 1.5]
        ax.set_yticks(yticks)
        ax.set_yticklabels([""] + [str(t) for t in yticks[1:]])  # no label for 0
        ax.set_xticklabels([])
        # ax.set_xticks(x)
        # ax.set_xticklabels(hormones, rotation=0, ha="center",fontsize=8,fontweight="bold", fontfamily="Arial")

       
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylabel(sex)
        for i, h in enumerate(hormones):
           dominant_mean = results[results['hormone'] == h]["mean_dominant"].iloc[0]
           submissive_mean = results[results['hormone'] == h]["mean_submissive"].iloc[0]

           y_line = 1.2   # line connecting dominant & submissive
           y_text = 1.3   # alpha text above the line

         # pick where to place the text depending on sex
           if sex == "female":
            #y_pos = max(dominant_mean, submissive_mean) + 0.5   # above bars
            ax.text(x[i],y_text + 0.10 ,  h,ha="center",va="bottom",fontsize=8,fontweight="bold",fontfamily="Arial" ,rotation = 90)

           
            va = "top"
           else:
            #y_pos =max(dominant_mean, submissive_mean) + 0.5  
                                             # below bars                                 
            va = "top"

           ax.plot([x[i] - bar_width / 2, x[i] + bar_width / 2],[y_line, y_line],linewidth=2, color = "black")

           ax.text(x[i], y_text, alphas[i],ha="center", va=va,fontsize=14, fontweight="bold")
        
        if sex == "female":
          

           dom_patch = Patch(facecolor="grey", alpha=1.0, label="Dominant")
           sub_patch = Patch(facecolor="grey", alpha=0.4, label="Submissive")
           ax.legend(handles=[dom_patch, sub_patch], frameon=False, fontsize=10, loc="upper left",bbox_to_anchor=(1.02, 1.0))


    #     #add raw data
        self.add_scatter_points(data,ax, hormones,sex,bar_width)
        
    #     plt.axvline(0, color='black',linewidth=1)  # vertical line at 0 (like in your PDF)
    #     plt.xlim(-1.03, 1.03)              
    #     ax.invert_yaxis()
        
        # plt.savefig("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/Difference_dom_sub_vs2" +sex + ".pdf", bbox_inches="tight")
        # plt.show()
    
    def add_scatter_points(self,data, ax, hormones,sex,bar_width):
        df = data
        x = np.arange(len(hormones))
        # scatter parameters
        rng = np.random.default_rng(42)  # reproducible jitter
        jitter_strength = 0.08
        
        
        if sex == "male":
            colord = "mediumaquamarine"
            colors = "aquamarine"
            edgecolor = "darkgreen"
        else:
            colord = "mediumpurple"
            colors = "blueviolet"
            edgecolor = "indigo"
        
        for i, hormone in enumerate(hormones):
          dom_vals = df.loc[df["Hierarchy"]=="dominant", hormone]
          sub_vals = df.loc[df["Hierarchy"]=="submissive", hormone]
          dom_x = np.full(len(dom_vals), x[i]- bar_width/2) + rng.normal(0, jitter_strength, size=len(dom_vals))
          sub_x= np.full(len(sub_vals), x[i]+bar_width/2) + rng.normal(0, jitter_strength, size=len(sub_vals))
          ax.scatter(dom_x, dom_vals,
                    facecolors=colord, edgecolors=edgecolor,
                    alpha=0.7, s=9, linewidths=1.2, zorder=2)
    
          ax.scatter(sub_x, sub_vals,
                     facecolors=colors, edgecolors=edgecolor,
                     alpha=0.6, s=9, linewidths=1.2, zorder=2)
   
    def normalize_data(self,data):
        df_normalized = data.copy()
        df_normalized.iloc[:,1:] = (data.iloc[:,1:] - data.iloc[:,1:].min()) / (data.iloc[:,1:].max() - data.iloc[:,1:].min())
        return df_normalized
    
    def plot_bar_plots_all(self,data_male, data_female, hormones,results_male, results_female):
       #fig, (ax_f, ax_m) = plt.subplots(2, 1, figsize=(15, 8),sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.01})
       fig, (ax_f, ax_m) = plt.subplots(2, 1, figsize=(20, 8),sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})

       #female panel
       self.plot_bar_plots(data_female,results_female,hormones,'female',ax_f)
       self.plot_bar_plots(data_male,results_male,hormones,'male',ax_m)
       ax_m.invert_yaxis()
       ax_m.xaxis.tick_top()
       ax_m.spines['top'].set_visible(True) 
       ax_m.spines['bottom'].set_visible(False) 
       ax_f.xaxis.tick_bottom()

       


       plt.savefig("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/Difference_dom_sub_all_biomarkers_vs1_fonts" + ".pdf", bbox_inches="tight")
       
       plt.show()


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
    
    
   

       



   ####################################################### 
    def __call__(self,hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        data_male = (self.data[self.data["sex"]== "male"]).iloc[:,1:]
        data_female = (self.data[self.data["sex"]== "female"]).iloc[:,1:]
        data_male_normalized = self.normalize_data(data_male)
        data_female_normalized = self.normalize_data(data_female)
        
        #without normalization
        results_male_wn, hormones_males_wn = self.compare_two_groups(data_male)
        results_female_wn, hormones_females_wn  = self.compare_two_groups(data_female)
        
        results_male, hormones_males = self.compare_two_groups(data_male_normalized)
        results_female, hormones_females  = self.compare_two_groups(data_female_normalized)
        hormones_to_consider = hormones_females_wn.tolist() + hormones_males_wn.tolist()
        hormones_to_consider = list(set(hormones_to_consider))

        hormones_to_consider1 = ['Corticosterone', 'T/Cort',  'Cort/DHEA', 'P/Cort', 'OEA/PEA', '1&2-AG/Cort','Asparagine/Aspartate','GABA', 'Glycine/Serine']
        hormones_to_consider_main = hormones + hormones_ratio + end_cann + end_cann_ratio + aminoacids + aminoacids_ratio

        hormones_to_consider = [h for h in hormones_to_consider_main if h not in hormones_to_consider1]

        self.plot_bar_plots_all(data_male_normalized, data_female_normalized, hormones_to_consider,results_male, results_female)

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
        # sex = "male"
       
        # self.plot_bar_plots(data_male_normalized,results_male,hormones_males,sex)
        
        # sex = "female"
        
        # self.plot_bar_plots(data_male_normalized,results_female,hormones_females,sex)
       
       # self.plot_bar_plots_all(data_male_normalized,data_male_normalized,results_male,results_male)
        
        a=1
