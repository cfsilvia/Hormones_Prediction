import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from scipy.optimize import linear_sum_assignment
import os

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
    def __call__(self):
        vectData = self.PC_comp.iloc[:, :3].to_numpy()
        vectArch_rows = self.Arch_comp.to_numpy()
        vectArch =vectArch_rows.T # transpose to shape(3,4)
        #create weights for each data
        W = create_personality.compute_barycentric_coords(vectData, vectArch_rows)
        #create dataframe and save with data information
        result, df = self.save_data(W)
        #do assignment according hungarian algorithm
        # df_assignment= self.hungarian_assignment(df)
        # #do assignment by taking acount the sex
        # df_assignment_Sex = self.hungarian_assignment_sex(result_arch,df)

        # result = pd.concat([self.SexTypeRank, df_assignment,df_assignment_Sex[df_assignment_Sex.columns[-1]]], axis=1)
        # result.to_excel(self.output_dir + 'assignment.xlsx', index=False)
        #ADD DATA TO THE HORMONES INFORMATION
        merged = self.add_status_to_hormones(result)
        return merged

    def add_status_to_hormones(self, result):
        #select given data
        selected_data = result
        merged = pd.merge(self.hormones_data, selected_data, on=['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy'], how='left')
        merged.to_excel(self.output_dir + 'data_for_model_with_biomarkers.xlsx', index=False)
        return merged 
    
    '''
    Sort according to the archetype we want to compare
    '''
    def save_data(self,W):
        arch_names = ["Arch1", "Arch2", "Arch3", "Arch4"]
        df = pd.DataFrame(W, columns=arch_names)
        #concatenate with data information
        result = pd.concat([self.SexTypeRank, df], axis=1)
       
        
        self.save_to_excel(result,(self.output_dir + 'weights_for_each_archetype.xlsx'), 'raw_data')
       
        return result, df
    
    '''
    do equal assignment according to the hungarian algorithm
    Creates 59 arch “positions”

     Builds a matrix of negative probabilities

     Hungarian finds assignment minimizing total cost

     This equals maximizing total probability

     Guarantees exact:
    '''
    def hungarian_assignment(self, df):
        df_copy = df.copy()
        arch_cols = df_copy.columns.tolist()
        #Define capacities how many data points we want to assign to each archetype
        capacities = {"Arch1": 15, "Arch2": 15, "Arch3": 15, "Arch4": 14}
        #Create cost matrix for the assignment problems
        slot_arches = []
        for arch, cap in capacities.items():
            slot_arches.extend([arch] * cap)  
        
        slot_arches = np.array(slot_arches)
        cost_matrix = np.zeros((len(df_copy), len(slot_arches)))

        for i, arch in enumerate(slot_arches):
            # negative because Hungarian minimizes cost
            cost_matrix[:, i] = -df[arch].values
        
        #Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        df_copy['assigned_arch'] = slot_arches[col_ind]

        return df_copy

    '''
    do assignment according to the hungarian algorithm but taking into account the sex
    '''
    def hungarian_assignment_sex(self, result, df):
        df_copy = df.copy()
        arch_cols = df_copy.columns.tolist()
        result_copy = result.copy()

        sex_counts = result_copy['sex'].value_counts()
        n_male =sex_counts.get('male', 0)
        n_female =sex_counts.get('female', 0)
        n_arch = len(arch_cols)

        #distribute
        male_distribution = create_personality.distribute(n_male, n_arch)
        female_distribution = create_personality.distribute(n_female, n_arch)

        slot_arch = []
        slot_sex = []

        for i, arch in enumerate(arch_cols):
            for _ in range(male_distribution[i]):
              slot_arch.append(arch)
              slot_sex.append("male")
            for _ in range(female_distribution[i]):
               slot_arch.append(arch)
               slot_sex.append("female")

        slot_arch = np.array(slot_arch)
        slot_sex = np.array(slot_sex)
        
        n_rows = len(df_copy)
        n_slots = len(slot_arch)
        cost_matrix = np.zeros((n_rows, n_slots))

        for i in range(n_rows):
            for j in range(n_slots):
                # negative because Hungarian minimizes cost
                arch = slot_arch[j]
                required_sex= slot_sex[j]
                if result_copy.loc[i, 'sex'] == required_sex:
                    cost_matrix[i, j] = -df_copy.loc[i, arch]
                else:
                    cost_matrix[i, j] = np.inf  # prohibit assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        df_copy['assigned_arch_sex'] = slot_arch[col_ind]
        
      
        return df_copy
  
    '''

    '''
    def save_to_excel(self, df, file_path, sheetname):
        # Save to Excel: create or append as needed
      if os.path.exists(file_path):  
        with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False, header = True)
      else:
        with pd.ExcelWriter(file_path, mode="w", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False, header =True) 

    


       




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
    

    '''
      input: X is (59x3) data , A archetypes coordinates as (4x3)
       output: weights for each data as function of arc - solve x=ATw,with  constraint ∑w=1
    '''
    @staticmethod
    def compute_barycentric_coords(X, A):
         N = X.shape[0]
         W = np.zeros((N, 4))

         # Augment archetypes matrix
         A_aug = np.vstack([A.T, np.ones(4)])  # shape (4,4) add a rows of one that is the constraint
         
         for i, x in enumerate(X):  
             x_aug = np.append(x, 1)  # shape (4,)
             w = np.linalg.solve(A_aug, x_aug)
             W[i] = w

         return W

   
    '''
    auxiliary function for equally distribution
    '''
    def distribute(total, groups):
       base = total // groups
       remainder = total % groups
       return [base + (1 if i < remainder else 0) for i in range(groups)]
    

    