import pandas as pd
import numpy as np

#THIS CLASS MANAGE HOW TO GET AN EXCEL FILE WITH ALL THE DATA
class manage_data_AviCondition:
    def __init__(self,pareto_file,pca_file,output_file,distance):
        self.pareto_file = pareto_file
        self.pca_file = pca_file
        self.output_file = output_file
        self.distance = distance
        
    '''
    input: pareto and pca files
    output: merge of pareto and pca files and take relevant data
    '''
    def append_data(self):
        #get architype components
        data_pareto = pd.read_excel(self.pareto_file, 'ArchInPcComp',header =None)
        data_pca =manage_data_AviCondition.read_excel_sheet(self.pca_file, 'Mean_Data')
        data_pca_components = manage_data_AviCondition.read_excel_sheet(self.pca_file, 'PCA_Data')
        #retain relevant inf. 
        columns_to_extract = [0,1,2,3,4,5,6,64,65,66,67,68,69,70] 
        data_pca_selected = data_pca.iloc[:,columns_to_extract]
        #convert into numpy matrix architypes in PC spac
        matrix = data_pareto.to_numpy().T
        row_of_ones = np.ones((1, matrix.shape[1]))
        matrix_with_ones = np.vstack([matrix, row_of_ones])
        
        #merge
        merged_df = pd.merge(data_pca_selected,data_pca_components, on = ['Experiment','sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal'])
        # Replace '/' with '_' in column names
        merged_df.columns = merged_df.columns.str.replace("/", "_", regex=False)
        
        return merged_df , matrix_with_ones     
       
    '''
    input: merged data
    output: add a status column
    '''
    def add_status(self, merged_df):
        merged_df['status'] = merged_df['Hierarchy']
        merged_df.loc[merged_df['Hierarchy'] != 'alpha','status'] = 'submissive'
        return merged_df
    '''
    input: merged_data
    output: normalize each vertix
    '''
    def normalize_distance(self,merged_df):
        columns_to_normalize = ["I", "A", "B", "P"]
        new_columns_names = [str(item) + '_normalized' for item in columns_to_normalize]
        merged_df[ new_columns_names] = merged_df[columns_to_normalize].apply (lambda x: (x - x.min()) / (x.max() - x.min()))
        return merged_df
    '''
   input:data
   output : data plus status for each architype
    '''
    def add_status_architype(self,merged_df):
        columns_to_normalize = ["I", "A", "B", "P"]
        for item in columns_to_normalize:
           column_name =  str(item) + '_normalized'
           column_name_new =  str(item) + '_status' 
           merged_df[column_name_new] = item
           merged_df.loc[merged_df[column_name] > self.distance,column_name_new] = 'O'
        return merged_df
    '''
    input:data with PC, with architype matrix
    output : thetus for each architype
    '''
    def  add_thetus(self, merged_df,matrix):
        solution_vectors = []
        columns_to_extract = ["PC1","PC2","PC3"]
        matrix = matrix.astype(float)
        for index, row in merged_df.iterrows():
            vector = row[columns_to_extract].to_numpy()
            vector = np.append(vector,1.0)
            vector =vector.astype(float)
            # Solve the system of equations
            x = np.linalg.solve(matrix, vector)
            solution_vectors.append(x)
        #stack
        data =np.vstack(solution_vectors)
        #create dataframe
        df = pd.DataFrame(data, columns =[ 'I','A','B','P'])   
        a=1 
            
            
            
           
    
    def __call__(self):
         merged_df , matrix_with_ones  = self.append_data()
         self.add_thetus(merged_df,matrix_with_ones)
         merged_df = self.add_status(merged_df)
         merged_df = self.normalize_distance(merged_df)
         merged_df = self.add_status_architype(merged_df)
         #save into  excel
         merged_df.to_excel("AviCond" + self.output_file, sheet_name = "All_data", index=False)
         a=1
         
   
    
    
        
    @staticmethod
    def read_excel_sheet(file_path, sheet_name):
            try:
                data = pd.read_excel(file_path, sheet_name=sheet_name)
                return data
            except Exception as e:
                print(f"Error reading the Excel sheet: {e}")
                return None