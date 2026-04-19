
import pandas as pd

class create_personality_second_method:
    def __init__(self,file_pareto,output_dir, hormones_file):
        self.weights = pd.read_excel(file_pareto,sheet_name = "Full_Data")
        self.output_dir = output_dir
        self.hormones_data = pd.read_excel(hormones_file, sheet_name = "All_data")

    '''
    General function
    '''
    def __call__(self):
      
        merged = self.add_status_to_hormones()
        return merged

    def add_status_to_hormones(self):
        #select given data
        selected_data = self.weights
        merged = pd.merge(self.hormones_data, selected_data, on=['Experiment',  'sex', 'Type',  'Genotype',  'Hierarchy','Mice.chips', 'Animal'], how='left')
        merged.to_excel(self.output_dir + 'data_for_model_with_biomarkers.xlsx', index=False)
        return merged 
    
   