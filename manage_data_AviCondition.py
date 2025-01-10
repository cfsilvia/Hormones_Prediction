import pandas as pd
import numpy as np

#THIS CLASS MANAGE HOW TO GET AN EXCEL FILE WITH ALL THE DATA
class manage_data_AviCondition:
    def __init__(self,pareto_file,pca_file):
        self.pareto_file = pareto_file
        self.pca_file = pca_file
        
       
        
    '''
    input: pareto and pca files
    output: merge of pareto and pca files and take relevant data
    '''
    def get_data(self):
        #get architype components
        data_pareto = pd.read_excel(self.pareto_file, 'ArchInPcComp',header =None)
       # data_pca =manage_data_AviCondition.read_excel_sheet(self.pca_file, 'Mean_Data')
        data_pca_components = manage_data_AviCondition.read_excel_sheet(self.pca_file, 'PCA_Data')
        #retain relevant inf. 
       # columns_to_extract = [0,1,2,3,4,5,6,64,65,66,67,68,69,70] 
        #data_pca_selected = data_pca.iloc[:,columns_to_extract]
        #convert into numpy matrix architypes in PC spac
        matrix = data_pareto.to_numpy().T
        row_of_ones = np.ones((1, matrix.shape[1]))
        matrix_with_ones = np.vstack([matrix, row_of_ones])

        return data_pca_components , matrix_with_ones     
       
   
    '''
    input:data with PC, with architype matrix
    output : thetus for each architype
    '''
    def  add_thetus(self, data_pca_components,matrix):
        solution_vectors = []
        columns_to_extract = ["PC1","PC2","PC3"]
        matrix = matrix.astype(float)
        for index, row in data_pca_components.iterrows():
            vector = row[columns_to_extract].to_numpy()
            vector = np.append(vector,1.0)
            vector =vector.astype(float)
            # Solve the system of equations
            x = np.linalg.solve(matrix, vector)
            vector_computed = matrix @ x
            
            solution_vectors.append(x)
        #stack
        data =np.vstack(solution_vectors)
        #create dataframe
        coef = pd.DataFrame(data, columns =[ 'I_avi','A_avi','B_avi','P_avi'])  
        data_pca_subset = data_pca_components.iloc[:, :7]
        combined_df = pd.concat([data_pca_subset, coef], axis =1) 
        return combined_df
            

    def __call__(self,data):
         df , matrix_with_ones  = self.get_data()
         combined_df = self.add_thetus(df,matrix_with_ones)
         merged_df = pd.merge(data,combined_df, on = ['Experiment','sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal'])
         
         return merged_df
         
   
    
    
        
    @staticmethod
    def read_excel_sheet(file_path, sheet_name):
            try:
                data = pd.read_excel(file_path, sheet_name=sheet_name)
                return data
            except Exception as e:
                print(f"Error reading the Excel sheet: {e}")
                return None