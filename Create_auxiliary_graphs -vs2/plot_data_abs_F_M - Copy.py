import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

class plot_data_abs_F_M:
    def __init__(self, data):
        self.data = data

    def __call__(self,hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        self.create_abs_list()
        mean_z_score, std_z_score, list_columns = self.order_abs_list(hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
        self.data_ordered = self.z_score.loc[:,list_columns]
        a=1

        
    '''
    select male, calculate mean and std, remove with this parameters mean and std
    from females
    '''
    def create_abs_list(self):
        male_data = self.data.loc[self.data["sex"] == "male"]
        male_data_mean = male_data.iloc[:, 1:].mean()
        male_data_std= male_data.iloc[:, 1:].std()

        female_data = self.data.loc[self.data["sex"] == "female"]
        # z_score_aux= female_data.iloc[:,1:] - male_data_mean
        z_score_aux= (female_data.iloc[:,1:]/male_data_mean)*100
        #self.z_score = z_score_aux/male_data_std
        self.z_score = z_score_aux - 100

    '''
    order list- for the three catergories and join in a new dataframe
    '''
    def order_abs_list(self, hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio):
        list_columns = []
        mean_z_score = self.z_score.mean()
        std_z_score = self.z_score.std()

        mean_hormones = (mean_z_score.loc[(hormones + hormones_ratio)]).sort_values(ascending = False)

        mean_end_cann = (mean_z_score.loc[(end_cann + end_cann_ratio)]).sort_values(ascending = False)
       
        mean_aminoacids = (mean_z_score.loc[(aminoacids + aminoacids_ratio)]).sort_values(ascending = False)
        list_columns = mean_hormones.index.tolist() + mean_end_cann.index.tolist() + mean_aminoacids.index.tolist()

      

        return mean_z_score, std_z_score, list_columns
