import pandas as pd
#THIS CLASS MANAGE HOW TO GET AN EXCEL FILE WITH ALL THE DATA
class manage_data:
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
        data_pareto = manage_data.read_excel_sheet(self.pareto_file, 'WithDistanceFromVertix')
        data_pca =manage_data.read_excel_sheet(self.pca_file, 'Mean_Data')
        #retain relevant inf. 
        columns_to_extract = [0,1,2,3,4,5,6,64,65,66,67,68,69,70] 
        data_pca_selected = data_pca.iloc[:,columns_to_extract]
        columns_to_extract = [0,1,2,3,4,5,6,10,11,12,13] 
        data_pareto_selected = data_pareto.iloc[:,columns_to_extract]
        #rename
        columns_to_rename = {7: 'I', 8: 'A' , 9: 'B' , 10: 'P'}  # Map index to new name
        data_pareto_selected.rename(columns={data_pareto_selected.columns[k]: v for k, v in columns_to_rename.items()}, inplace=True)
        #merge
        merged_df = pd.merge(data_pca_selected,data_pareto_selected, on = ['Experiment','sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal'])
        
        return(merged_df)       
       
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
        
    
    def __call__(self):
         merged_df = self.append_data()
         merged_df = self.add_status(merged_df)
         merged_df = self.normalize_distance(merged_df)
         merged_df = self.add_status_architype(merged_df)
         #save into  excel
         merged_df.to_excel(self.output_file, sheet_name = "All_data", index=False)
         a=1
         
   
    
    
        
    @staticmethod
    def read_excel_sheet(file_path, sheet_name):
            try:
                data = pd.read_excel(file_path, sheet_name=sheet_name)
                return data
            except Exception as e:
                print(f"Error reading the Excel sheet: {e}")
                return None