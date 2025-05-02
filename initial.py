from manage_data import manage_data
from manage_data_AviCondition import manage_data_AviCondition
from treat_data import treat_data
import pandas as pd
import pickle
from plot_data import plot_data
from treat_validation_data import treat_validation_data
from Find_better_features import Find_better_features
from treat_random_data import treat_random_data
from General_functions import General_functions
import yaml
from memory_profiler import profile
import Auxiliary_functions
import os
#from sklearn.ensemble import AdaBoostClassifier
@profile
def main_menu(choice,data):
     
      
            
        if choice == "1":
            type = data['1']['type'] 
            input_file =data['1']['data_file']
            sex = data['1']['sex'] #female or all
            n_repeats = data['1']['n_repeats']
            hormones = data['1']['hormones']
            output_directory = data['1']['output_directory']
            list_models = data['1']['models']
            normalization = data['1']['normalization']
            select_pairs = data['1']['select_pairs']
        
            for  p in select_pairs:
                title_file = sex + "_" + type + "_" + '_'.join(p)   
                print('_'.join(p) )   
                for model in list_models:
                        model_dict ={}
                        new_obj = treat_data(input_file,p)
                        results_dict = new_obj(model, normalization, n_repeats,sex,hormones)
                        print(model)
                        model_dict[model] = results_dict # for each model there is a dictionary
                        filename = output_directory + title_file + '.pkl'
                        Auxiliary_functions.save_part_of_dict(filename, model, model_dict) #for each pairs save a pkl function
                Auxiliary_functions.save_as_excel(output_directory,title_file,len(p))
              
        elif choice == "2": 
            type = data['2']['type']
            sex = data['2']['sex']
            n_repeats = data['2']['n_repeats']
            ouput_directory = data['2']['output_directory']
            select_pairs = data['2']['select_pairs']
            model_name = data['2']['model_name']
            
            total_data_final = pd.DataFrame()
            total_data_before_final = pd.DataFrame()
            for  p in select_pairs:
                title_file = sex + "_" + type + "_" + '_'.join(p)  
                # Load the Pickle file
                with open(ouput_directory + title_file + '.pkl', "rb") as f:
                   data = pickle.load(f)
                
                new_obj = plot_data(data, title_file, ouput_directory,sex)
                new_obj(len(p),model_name)
                # total_data_final = pd.concat([total_data_final, total_data], axis=0)
              


if __name__ == "__main__":
    with open("F:/SilviaData/rutiFrishman/settings_windows_last_version.yml", "r") as file: #CHANGE WHEN NECCESSARY DIRECTORY OF SETTINGS
        data = yaml.safe_load(file)
    choice = data['choice']    
    main_menu(choice,data)