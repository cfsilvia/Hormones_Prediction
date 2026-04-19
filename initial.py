from manage_data import manage_data
from create_personality import create_personality
from create_personality_second_method import create_personality_second_method
from treat_data import treat_data
#from treat_data_personality import treat_data_personality
import pandas as pd
import pickle
from plot_data import plot_data
from plot_data_personality import plot_data_personality
from treat_validation_data import treat_validation_data
from Find_better_features import Find_better_features
from treat_continous_labels_1 import treat_continous_labels
from treat_continous_labels_1_second_method import treat_continous_labels_second_method
from General_functions import General_functions
from statistics_class import statistics_class
import yaml
from memory_profiler import profile
import Auxiliary_functions
import os
from scipy.stats import mannwhitneyu
from Pareto import PCHA


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
            type_graph = data['2']['type_graph']
            
            total_data_final = pd.DataFrame()
            total_data_before_final = pd.DataFrame()
            for  p in select_pairs:
                title_file = sex + "_" + type + "_" + '_'.join(p)  
                # Load the Pickle file
                with open(ouput_directory + title_file + '.pkl', "rb") as f:
                   data = pickle.load(f)
                
                new_obj = plot_data(data, title_file, ouput_directory,sex)
                new_obj(len(p),model_name,type_graph)
                # total_data_final = pd.concat([total_data_final, total_data], axis=0)
         
        # elif choice == "3": #create the status of personality
        #      file_pareto = data['3']['data_pareto']
        #      output_dir = data['3']['output_dir']
        #      hormones_file = data['3']['hormones_file']
        #      list_models = data['3']['models']
        #      normalization = data['3']['normalization']
        #      select_pairs = data['3']['select_pairs']
        #      hormones = data['3']['hormones']
        #      sex = data['3']['sex'] #female or all
        #      n_repeats = data['3']['n_repeats']
        #      type_personality = data['3']['type_personality']
        #      create_table = data['3']['create_table']
        #      run_model = data['3']['run_model']

        #      if create_table == True:
        #        new_obj = create_personality(file_pareto,output_dir, hormones_file)
        #        data_to_predict = new_obj() #create the table with the personality status for each data point

        #      if run_model:  
             


        #       for  p in select_pairs:
        #         data_to_predict = pd.read_excel((output_dir + 'data_for_model_with_assignment.xlsx'))

        #         title_file = sex +  '_'.join(p)   
        #         print('_'.join(p) )   
        #         for model in list_models:
        #                 model_dict ={}
        #                 new_obj = treat_data_personality(data_to_predict,p)
        #                 results_dict = new_obj(model, normalization, n_repeats,sex,hormones)
        #                 print(model)
        #                 model_dict[model] = results_dict # for each model there is a dictionary
        #                 filename = output_dir + title_file + '.pkl'
        #                 Auxiliary_functions.save_part_of_dict(filename, model, model_dict)
        #         Auxiliary_functions.save_as_excel(output_dir,title_file,len(p))

        elif choice == "4": 
            
            sex = data['4']['sex']
            n_repeats = data['4']['n_repeats']
            ouput_directory = data['4']['output_directory']
            select_pairs = data['4']['select_pairs']
            model_name = data['4']['model_name']
            type_graph = data['4']['type_graph']
            
            total_data_final = pd.DataFrame()
            total_data_before_final = pd.DataFrame()
            for  p in select_pairs:
                title_file = sex + '_'.join(p)  
                # Load the Pickle file

                
                with open(ouput_directory + title_file + '.pkl', "rb") as f:
                   data = pickle.load(f)
                
                new_obj = plot_data_personality(data, title_file, ouput_directory,sex)
                new_obj(len(p),model_name,type_graph)
        
        elif choice == "5":
            input_file = data['5']['input_file']
            initial = data['5']['initial_excel_column_consider']
            final = data['5']['final_excel_column_consider']





            obj_statistics = statistics_class(input_file, initial, final)
            obj_statistics()
   
              
        elif choice == "6":
           file_pareto = data['6']['data_pareto']
           output_dir = data['6']['output_dir']
           hormones_file = data['6']['hormones_file']
           create_table = data['6']['create_table']
           run_model = data['6']['run_model']
           run_correlation = data['6']['run_correlation']
           run_features_normalization = data['6']['run_features_normalization']
           type_model = data['6']['type_model']
           external_file = data['6']['external_file']
           compare_with_baseline = data['6']['compare_with_baseline']




           if create_table:
                new_obj = create_personality(file_pareto,output_dir, hormones_file)
                #new_obj.add_status_to_hormones_from_external_file(external_file)
                data_to_predict = new_obj() #create the table with the personality status for each data point

           if run_model:  

                new_obj = treat_continous_labels(output_dir + 'data_for_model_with_biomarkers.xlsx',output_dir,run_features_normalization, type_model, compare_with_baseline)
                new_obj()

           if run_correlation:
                new_obj = General_functions(output_dir + 'data_for_model_with_biomarkers.xlsx',output_dir)
                new_obj()

        elif choice == "7": 
                input_file = data['7']['input_file']
                output_dir = data['7']['output_dir']
                
                # Load your PCA data
                df = pd.read_excel(input_file)
                X = df.values

                model = PCHA(k=4, n_iter=50)
                model.fit(X, compute_pvalues=True)

                model.save_to_excel( output_dir + 'pcha_results.xlsx'
                                    )
                model.plot_3d()
                # Save results
               # model.save()
        elif choice == "8": #use in the data with 3 archetypes
           file_pareto = data['8']['data_cluster']
           output_dir = data['8']['output_dir']
           hormones_file = data['8']['hormones_file']
           create_table = data['8']['create_table']
           run_model = data['8']['run_model']
           
           run_features_normalization = data['8']['run_features_normalization']
           type_model = data['8']['type_model']
           compare_with_baseline = data['8']['compare_with_baseline']


           if create_table:
                new_obj = create_personality_second_method(file_pareto,output_dir, hormones_file)
                data_to_predict = new_obj() #create the table with the personality status for each data point

           if run_model:  

                new_obj = treat_continous_labels_second_method(output_dir + 'data_for_model_with_biomarkers.xlsx',output_dir,run_features_normalization, type_model, compare_with_baseline)
                new_obj()


     
                
if __name__ == "__main__":

    with open("U:/Users/Silvia/RutiFrishman_2025_hormones_paper/settings_windows_last_version_october_2025.yml", "r") as file: #CHANGE WHEN NECCESSARY DIRECTORY OF SETTINGS
        data = yaml.safe_load(file)
    choice = data['choice']    

    main_menu(choice,data)