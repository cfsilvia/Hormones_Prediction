import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,        # TrueType fonts
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
import matplotlib.pyplot as plt 
from scipy.stats import sem


class plot_data_abs_F_M:
    def __init__(self, output_dir, data):
        self.data = data
        self.output_dir = output_dir

    def __call__(self,hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        self.create_abs_list()
        list_columns = self.order_abs_list(hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
        self.percent_ordered = self.percent_diff.loc[list_columns]
        self.se_percent_ordered = self.se_percent.loc[list_columns]
        
        len_hormones = len(hormones+hormones_ratio)
        len_endocann = len(end_cann + end_cann_ratio)
        len_aminoac = len(aminoacids + aminoacids_ratio)
        self.plot_bar_plot( len_hormones, len_endocann, len_aminoac)
        
        a=1

        
    '''
    select male, calculate mean and std, remove with this parameters mean and std
    from females
    '''
    def create_abs_list(self):
        male_data = self.data.loc[self.data["sex"] == "male"]
        male_data_mean = male_data.iloc[:, 1:].mean()
        male_data_std= male_data.iloc[:, 1:].apply(sem)
        
        female_data = self.data.loc[self.data["sex"] == "female"]
        female_data_mean = female_data.iloc[:, 1:].mean()
        female_data_std= female_data.iloc[:, 1:].apply(sem)

        # Percent difference (Female relative to Male)
        self.percent_diff = 100 * (female_data_mean - male_data_mean) / male_data_mean

        self.se_percent = np.sqrt((100 / male_data_mean)**2 * female_data_std**2 + (100 * female_data_mean / male_data_mean**2)**2 * male_data_std**2)

        


        

    '''
    order list- for the three catergories and join in a new dataframe
    '''
    def order_abs_list(self, hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        list_columns = []
        

        mean_hormones = (self.percent_diff.loc[(hormones + hormones_ratio)]).sort_values(ascending = False)

        mean_end_cann = (self.percent_diff.loc[(end_cann + end_cann_ratio)]).sort_values(ascending = False)
       
        mean_aminoacids = (self.percent_diff.loc[(aminoacids + aminoacids_ratio)]).sort_values(ascending = False)
        list_columns = mean_hormones.index.tolist() + mean_end_cann.index.tolist() + mean_aminoacids.index.tolist()

      

        return  list_columns
    
    '''
    plot bar plots
    '''
    def plot_bar_plot(self, len_hormones, len_endocann, len_aminoac):
       

       #x positions
       x = np.arange(self.percent_ordered.shape[0])
       #define colors
       colors = [""]*self.percent_ordered.shape[0]

       colors[:len_hormones] = ["red"] * len_hormones
       colors[len_hormones:(len_endocann + len_hormones)] = ["green"] * len_endocann
       colors[(len_endocann + len_hormones):] = ["purple"] * len_aminoac

       #plot
       fig,ax = plt.subplots(figsize = (12,6))

       ax.bar(x, self.percent_ordered, yerr = self.se_percent_ordered, capsize = 3, color =colors, width = 1.0, linewidth =0, alpha = 0.8)

       # baseline and style
       ax.axhline(0, color="black", linewidth=1)
       ax.set_xticks(x)
       ax.set_xticklabels(self.percent_ordered.index, rotation=60, ha="right", fontsize = 10)
       ax.set_ylabel("Female − Male (%)")
       ax.set_ylim(-100,100)

        # light grid, like your example
       #ax.grid(axis="y", linestyle="--", alpha=0.4)
       ax.grid(False)

       plt.tight_layout()
       
       plt.savefig(self.output_dir + "delta_f_m_fonts.pdf", format = "pdf", bbox_inches = "tight")

       plt.show()
