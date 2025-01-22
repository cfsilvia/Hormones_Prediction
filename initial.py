from manage_data import manage_data
from manage_data_AviCondition import manage_data_AviCondition
from treat_data import treat_data
import pandas as pd
import pickle
from plot_data import plot_data
from treat_validation_data import treat_validation_data
from Find_better_features import Find_better_features
from treat_random_data import treat_random_data
import yaml
#from sklearn.ensemble import AdaBoostClassifier

def main_menu(choice,data):
     
        if choice == "1":
            pareto_file = data['1']['pareto_file'] 
            data_file = data['1']['data_file']
            output_file =data['1']['output_file']
            distance = data['1']['distance']
            columns_hormones_to_extract = data['1']['columns_to_extract']
            
            
            new_obj = manage_data(pareto_file, data_file, output_file, distance,columns_hormones_to_extract)
           
            new_obj()
            
        elif choice == "2":
            type = data['2']['type'] 
            output_file =data['2']['data_file']
            sex = data['2']['sex'] #female or all
            input_file = data['2']['data_features']
            n_repeats = data['2']['n_repeats']
            ouput_directory = data['2']['output_directory']
            randomization = False # in the case want to randomize the classification
            num_permutations = 3 
            
            hormones_combination = Find_better_features.get_best_combinations(input_file)
            title_file = sex + "_" + type + "_" + str(n_repeats)
           #for randomization
            #title_file = sex + "_" + type + "_" + str(n_repeats) + "_" + str(num_permutations)
            list_models = ["SVC_linear","SVC_rbf","random_forest","logistic", "decision_tree","k_neighbors","qda"]
            hormones_dict = {} #for each hormone dict there are different models dict and each one has the results in the form of dict
           
            if randomization:
                for count in range(len(hormones_combination)):
                    model_dict ={}
                    for model in list_models:
                        new_obj = treat_random_data(output_file)
                        results_dict = treat_random_data._call_(new_obj,model, n_repeats,num_permutations,sex,choice,hormones_combination[count])
                        model_dict[model] = results_dict # for each model there is a dictionary
                        key_hormones = "-".join(hormones_combination[count])
                        hormones_dict[key_hormones] = model_dict
                        a=1
                    
               
            else:
                #loop to ge several hormone combinations
                for count in range(len(hormones_combination)):
                    model_dict ={}
                    
                    for model in list_models:
                        new_obj = treat_data(output_file)
                        results_dict = new_obj(model, n_repeats,sex,choice,hormones_combination[count])
                        model_dict[model] = results_dict # for each model there is a dictionary
                        key_hormones = "-".join(hormones_combination[count])
                        hormones_dict[key_hormones] = model_dict
                        
                    
            # # Save the dictionary
            with open(ouput_directory + title_file + '.pkl', 'wb') as f:
                pickle.dump(hormones_dict, f)
                
     
        
        
        elif choice == "3":
            type = data['3']['type']
            output_file =data['3']['data_file']
            sex = data['3']['sex']
            input_file = data['3']['data_features']
            ouput_directory = data['3']['output_directory']
            n_repeats = data['3']['n_repeats']
             
            hormones_combination = Find_better_features.get_best_combinations(input_file)
            title_file = sex + "_" + type + "_" + str(n_repeats)
            #list_models = ["SVC_linear","SVC_rbf","random_forest","logistic","GaussianNB"]
            list_models = ["SVC_linear","SVC_rbf","random_forest","logistic", "decision_tree","k_neighbors","qda"]
            hormones_dict = {} #for each hormone dict there are different models dict and each one has the results in the form of dict
           
            #loop to ge several hormone combinations
            for count in range(len(hormones_combination)):
                model_dict ={}
                
                for model in list_models:
                    new_obj = treat_data(output_file)
                    results_dict = new_obj(model, n_repeats,sex,choice,hormones_combination[count],type)
                    model_dict[model] = results_dict # for each model there is a dictionary
                    key_hormones = "-".join(hormones_combination[count])
                    hormones_dict[key_hormones] = model_dict
                    
            # # Save the dictionary
            with open(ouput_directory + title_file + '.pkl', 'wb') as f:
                pickle.dump(hormones_dict, f)
                

              
        elif choice == "4": 
            type = data['4']['type']
            sex = data['4']['sex']
            n_repeats = data['4']['n_repeats']
            ouput_directory = data['4']['output_directory']
            select_column_prob = 1
            title_file = sex + "_" + type + "_" + str(n_repeats)
            # Load the Pickle file
            with open(ouput_directory + title_file + '.pkl', "rb") as f:
                data = pickle.load(f)
                
            new_obj = plot_data(data, title_file, ouput_directory)
            total_data, total_data_filter_confusion,total_data_filter_all = new_obj(select_column_prob)
           
            
            with pd.ExcelWriter(ouput_directory + title_file  + '.xlsx') as writer:
                total_data.to_excel(writer, sheet_name='all_predictions', index=False)
                total_data_filter_confusion.to_excel(writer, sheet_name='all_confusion_filter', index=False)
                total_data_filter_all.to_excel(writer, sheet_name='final_data', index=False)

        elif choice == "5":
                 type = data['5']['type']
                 sex = data['5']['sex']
                 choice_m = data['5']['choice_m']
                 n_repeats = data['5']['n_repeats']
                 output_file =data['5']['data_file']
                 index_initial_hormone = data['5']['index_begin'] -1
                 index_final_hormone = data['5']['index_final'] 
                 ouput_directory = data['5']['output_directory'] 
                 features = {}
                 
                 data = pd.read_excel(output_file,sheet_name="All_data")
                 hormones =  data.iloc[:,index_initial_hormone:index_final_hormone].columns.tolist()
                 
                 #models_list = ["SVC_linear","random_forest","logistic"] "SVC_rbf" doesnt work
                 #models_list = ["SVC_linear","random_forest","logistic", "decision_tree","k_neighbors","adaboost","qda"]
                 models_list = ["SVC_linear","random_forest","logistic", "decision_tree"]
                 
                 positive_feature = "alpha"
                 title_file = 'Better_features' + '_' + sex + "_" + type + "_" + str(n_repeats)
                 
                 for model_name in models_list:
                    new_obj = Find_better_features(output_file, positive_feature)
                    final_features = new_obj(model_name,n_repeats, sex, choice_m, hormones,type)
                    features[model_name] = {'sex' : sex, 'type' : type , 'final_features' : final_features}
                    a=1
                    
                    # # Save the dictionary

                 with open(ouput_directory + title_file + '.pkl', 'wb') as f:
                    pickle.dump(features, f)
                    
        elif choice == "6": #for validation-do learning in all original file-and then check on the validation
            type = data['6']['type']
            train_file =data['6']['train_file']
            validation_file = data['6']['validation_file']
            sex = data['6']['sex']
            choice = data['6']['choice']
            n_repeats = data['6']['n_repeats']
            
            columns_validation = ['sex','status','Prog','T','Cort','DHEA','AEA','AG',
                                  'OEA','SEA','PEA']
            hormones_list = ['Prog-T-Cort_to_T']
            model = ["SVC_linear"]
            hormones_dict = {}
            
            for count, h in enumerate(hormones_list):
              #convert h into list
              hormones = h.split('-')
              new_obj = treat_validation_data(train_file, validation_file,
                                              columns_validation)
              results_dict = new_obj(model[count] ,n_repeats, sex , choice , hormones)
              hormones_dict[h] = results_dict
                 
      
        else:
            print("Invalid choice. Please try again.")




if __name__ == "__main__":
    with open("settings.yml", "r") as file:
        data = yaml.safe_load(file)
    choice = data['choice']    
    main_menu(choice,data)