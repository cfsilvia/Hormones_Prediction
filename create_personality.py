import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
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
    def __call__(self,pair_compare):
        vectData = self.PC_comp.iloc[:, :3].to_numpy()
        vectArch_rows = self.Arch_comp.to_numpy()
        vectArch =vectArch_rows.T # transpose to shape(3,4)
        #create weights for each data
        W, err = create_personality.nnls_weights(vectArch, vectData)
        #create dataframe and save with data information
        result = self.save_data(W, err, pair_compare)
        #ADD DATA TO THE HORMONES INFORMATION
        #merged = self.add_status_to_hormones(result)
        return result

    def add_status_to_hormones(self, result):
        #select given data
        selected_data = result[['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy','personality']]
        merged = pd.merge(self.hormones_data, selected_data, on=['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy'], how='left')
        merged.to_excel(self.output_dir + 'data_for_model.xlsx', index=False)
        return merged 
    
    '''
    Sort according to the archetype we want to compare
    '''
    def save_data(self,W, err, pairs):
        arch_names = ["Arch1", "Arch2", "Arch3", "Arch4"]
        df = pd.DataFrame(W, columns=arch_names)
        #concatenate with data information
        result = pd.concat([self.SexTypeRank, df], axis=1)
        # Sort indices in descending order for each row
        sorted_idx = np.argsort(W, axis=1)[:, ::-1]
        # Take the first largest (index 0)
        first_idx = sorted_idx[:, 0]
        result["personality"] = np.array(arch_names)[first_idx]
        result["personality%"] = (W[np.arange(W.shape[0]), first_idx] * 100).round(2)
        
        for p in pairs:
          result = self.process_until_empty( W, arch_names, result,p,sorted_idx)
        #filter according max/min >1.4
        result_filter = result.copy()
        for p in pairs:
          result_filter = self.filter(result_filter, p)
        
        self.save_to_excel(result,(self.output_dir + 'weights_all_pairs.xlsx'), 'first_assignment')
        self.save_to_excel(result_filter,(self.output_dir + 'weights_all_pairs.xlsx'), 'first_filter') 

        
        for p in pairs:
          result_second_filter = result_filter.copy()
          result_second_filter = self.filter_second(result_second_filter, p)
          result_second_filter_1 = result_second_filter.copy()
          result_second_filter_1['personality'] = result_second_filter[str(p)]
          result_second_filter_1['personality%'] = result_second_filter[(str(p) + '%')]
          result_second_filter_1 =result_second_filter_1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
          #add hormones
          merged = self.add_status_to_hormones(result_second_filter_1)

          self.save_to_excel(result_second_filter_1,(self.output_dir + 'weights_all_pairs.xlsx'), (str(p[0]) + '_' + str(p[1])))
          self.save_to_excel(merged,(self.output_dir + 'weights_all_pairs_with_hormones.xlsx'), (str(p[0]) + '_' + str(p[1])))

        #  #add the W influence more
        #  top_idx = np.argmax(W, axis=1)
        #  result["personality"] = np.array(arch_names)[top_idx]
        #  result["personality%"] = (W[np.arange(W.shape[0]), top_idx] * 100).round(2)
        # result["RMSE"] = err.round(4)

        #result.to_excel(self.output_dir + 'weights_all_pairs.xlsx', index=False)
        

        return merged
    '''
    '''
    def filter_second(self,df, p):
      ps = str(p)
      p_perc = (str(p) + '%')
      p_perc = str(p_perc)

      columns = df.columns.tolist()
      final_data = pd.DataFrame(columns = columns)
      
      #sorted
      df_sorted = df[df[ps] != False].sort_values(by = [ps, 'sex', p_perc], ascending = [True, True, False])

      for i in range(2):
        #selection
        aux_1 = df_sorted[(df_sorted['sex'] == 'female') & (df_sorted[ps] == p[i])].head(5)
        aux_2 = df_sorted[(df_sorted['sex'] == 'male') & (df_sorted[ps] == p[i])].head(5)
        
        # If one has fewer than 5, fill from the other
        if len(aux_1) < 5:
            needed = 5 - len(aux_1)
            aux = pd.concat([aux_1, df_sorted[(df_sorted['sex'] == 'male') & (df_sorted[ps] == p[i])].head((5 + needed))], ignore_index=True)
        elif len(aux_2) < 5:
            needed = 5 - len(aux_2)
            aux = pd.concat([aux_2, df_sorted[(df_sorted['sex'] == 'female') & (df_sorted[ps] == p[i])].head(5 + needed)], ignore_index=True)
        else:
           aux = pd.concat([aux_1, aux_2], ignore_index = True)

        final_data = pd.concat([final_data, aux], ignore_index = True)
      return final_data


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
    '''
    def filter(self, df, p):
        ratio = np.where((df[p[0]] == 0) | (df[p[1]] == 0), 100 , np.maximum(df[p[1]], df[p[0]]) / np.minimum(df[p[1]], df[p[0]]))
        df.loc[ratio < 1.4, str(p)] = False
        return df

    '''
    '''
    def process_until_empty(self, W, arch_names, result,p, sorted_idx):
        importance_index = 1
        result[str(p)] = result["personality"]
        result[(str(p) + '%')] = result["personality%"]
        _, index_rows = self.find_others(result, p)
        while index_rows is not None and len(index_rows) > 0:

            second_idx = sorted_idx[index_rows, importance_index]

            result.loc[index_rows,str(p)] = (np.array(arch_names)[second_idx])
            result.loc[index_rows,(str(p) + '%')] = (W[index_rows, second_idx] * 100).round(2)
            importance_index += 1
            _, index_rows = self.find_others(result, p)

        return result
    
    '''
    '''
    def find_others(self, df,p):
      a =p[0]
      b = p[1]
      # Exclude rows that contain either string
      mask = ~((df[str(p)].str.contains(a, na=False)) | (df[str(p)].str.contains(b, na=False)))
      result = df[mask]
      index_rows = (df.index[mask]).tolist()

      return  result, index_rows
   


       




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
    
   
