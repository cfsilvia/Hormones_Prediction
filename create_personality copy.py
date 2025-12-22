import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.optimize import lsq_linear

class create_personality:
    def __init__(self,file_pareto,output_dir, hormones_file):
        self.PC_comp = pd.read_excel(file_pareto,sheet_name="PcComponents", header = None)
        self.Arch_comp = pd.read_excel(file_pareto,sheet_name="ArchInPcComp", header = None)
        self.SexTypeRank = pd.read_excel(file_pareto,sheet_name="Sex-Type-Ranking", header = None)
        self.SexTypeRank.rename(columns={0: "Experiment", 1: "sex", 2: "Type", 3: "Genotype", 4: "Hierarchy"}, inplace=True)
        self.output_dir = output_dir
        self.hormones_data = pd.read_excel(hormones_file, sheet_name = "All_data")

    '''
    General function
    '''
    def __call__(self,type_pers):
        vectData = self.PC_comp.iloc[:, :3].to_numpy()
        vectArch_rows = self.Arch_comp.to_numpy()
        vectArch =vectArch_rows.T # transpose to shape(3,4)
        #create weights for each data
        W, err = create_personality.nnls_weights(vectArch, vectData)
        #create dataframe and save with data information
        result = self.save_data(W, err, type_pers)
        #ADD DATA TO THE HORMONES INFORMATION
        merged = self.add_status_to_hormones(result)
        return merged

    def add_status_to_hormones(self, result):
        #select given data
        selected_data = result[['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy','personality']]
        merged = pd.merge(self.hormones_data, selected_data, on=['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy'], how='left')
        merged.to_excel(self.output_dir + 'data_for_model.xlsx', index=False)
        return merged 

    def save_data(self,W, err,type):
        arch_names = ["Arch1", "Arch2", "Arch3", "Arch4"]
        df = pd.DataFrame(W, columns=arch_names)
        #concatenate with data information
        result = pd.concat([self.SexTypeRank, df], axis=1)
        if type == 2:
           # Sort indices in descending order for each row
           sorted_idx = np.argsort(W, axis=1)[:, ::-1]
           # Take the 2nd largest (index 1)
           second_idx = sorted_idx[:, 1]
           result["personality"] = np.array(arch_names)[second_idx]
           result["personality%"] = (W[np.arange(W.shape[0]), second_idx] * 100).round(2)
        elif type == 3:
           # Sort indices in descending order for each row
           sorted_idx = np.argsort(W, axis=1)[:, ::-1]
           # Take the 2nd largest (index 1)
           second_idx = sorted_idx[:, 2]
           result["personality"] = np.array(arch_names)[second_idx]
           result["personality%"] = (W[np.arange(W.shape[0]), second_idx] * 100).round(2)
        else:
         #add the W influence more
         top_idx = np.argmax(W, axis=1)
         result["personality"] = np.array(arch_names)[top_idx]
         result["personality%"] = (W[np.arange(W.shape[0]), top_idx] * 100).round(2)
        result["RMSE"] = err.round(4)

        result.to_excel(self.output_dir + 'weights.xlsx', index=False)

        return result
    

    '''
    Sort according to the archetype we want to compare
    '''


       




    '''
     input:PC of each point and Pc comp.of each othe archetype
     A is a 3x4 array in which each column are the PC components of each arch, V is an array in which each row is one data point in the PC comp
     The idea is fo find weights for each data point which are >0 and weights =1 such that 
     w≥0 min​∥Aw−v∥22​
     output: coefficients that relate each point with the 4 archetype
    '''
    @staticmethod
    def nnls_weights(A,V):
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1.0
        A_scaled = A / col_norms
       
        W = np.zeros((V.shape[0],A.shape[1]))
        for i,v in enumerate(V):
            v_scaled = v.copy()
            w_scaled, _ = nnls(A_scaled,v_scaled,maxiter=200) # w >= 0 minimizing ||A w - v||
            w = w_scaled / col_norms     # undo the scaling
            s = w.sum()
            W[i] = w / s if s > 0 else w #normalize to get the sum w=1
        
        recon = (A @ W.T).T
        err = np.linalg.norm(V - recon, axis=1)



        return W, err
    
   
