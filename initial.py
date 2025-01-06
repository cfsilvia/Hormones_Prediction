from manage_data import manage_data
from manage_data_AviCondition import manage_data_AviCondition
from treat_data import treat_data
import pandas as pd
import pickle
from plot_data import plot_data
from Find_better_features import Find_better_features
from treat_random_data import treat_random_data

def main_menu():
     while True:
        print("\nMain Menu")
        print("1. Option 1: First step, create an excel with the data to use")
        print("2. Option 2: Predict hierarchy")
        print("3. Option 3: Predict personality")
        print("4. Option 4: Get graphs and tables")
        print("5. Option 5: Find better features for each model")
        print("6. Option 6: break")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            pareto_file = 'F:/Ruti/Pareto/10.09.24_stage2_stage2_Pareto_Information.xlsx'
            data_file = 'F:/Ruti/PCA/PCA_PCA__10.9.24_stage2_NEW_stage2.xlsx'
            output_file ='F:\Ruti\AnalysisWithPython\data_to_use.xlsx'
            distance = 0.5
            
            #new_obj = manage_data(pareto_file, data_file, output_file, distance)
            new_obj = manage_data_AviCondition(pareto_file, data_file, output_file, distance)
            new_obj()
            
        elif choice == "2":
            type = "hierarchy_randomization"
            output_file ='F:\Ruti\AnalysisWithPython\data_to_use.xlsx'
            sex = 'male' #female or all
            input_file = 'F:/Ruti/AnalysisWithPython/Better_features/Better_features_male_hierarchy_100.pkl'
    #         hormones_combination =  [['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA'],['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio'], ['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA','Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio'],['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio'],
    #                                 ['Hair.T','Hair.T_Cort.ratio'],['Hair.P','Hair.P_Cort.ratio'],['Hair.DHEA','Hair.Cort_DHEA.ratio'],['Hair.Cort', 'Hair.Cort_DHEA.ratio'],
    #                                 ['Hair.Cort', 'Hair.DHEA', 'Hair.T_Cort.ratio', 'Hair.P_Cort.ratio',
    #    'Hair.Cort_DHEA.ratio']]
            hormones_combination = Find_better_features.get_best_combinations(input_file)
            hormones_combination.append(['Hair.T','Hair.T_Cort.ratio'])
            hormones_combination.append(['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio'])
            hormones_combination.append(['Hair.DHEA','Hair.Cort_DHEA.ratio'])
            hormones_combination.append(['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio'])
            
            #for females
            hormones_combination.append(['Hair.P','Hair.P_Cort.ratio'])
            hormones_combination.append(['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio'])
            hormones_combination.append(['Hair.T','Hair.T_Cort.ratio'])
           
            
            
            n_repeats = 100
            ouput_directory = "F:/Ruti/AnalysisWithPython/"
            randomization = True # in the case want to randomize the classification
            num_permutations = 3
           # title_file = sex + "_" + type + "_" + str(n_repeats)
           #for randomization
            title_file = sex + "_" + type + "_" + str(n_repeats) + "_" + str(num_permutations)
            list_models = ["SVC_linear","SVC_rbf","random_forest","logistic"]
            #list_models = ["SVC_linear","SVC_rbf","random_forest","logistic","GaussianNB"]
            total_results_probability = pd.DataFrame()
            total_results_accuracy = pd.DataFrame()
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
            type = "I_status"
            output_file ='F:\Ruti\AnalysisWithPython\data_to_use.xlsx'
            sex = 'all' #female or all
            input_file = 'F:\Ruti\AnalysisWithPython\Better_features\Better_features_all_I_status_100.pkl'
            # hormones_combination = [['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA'],['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio'], ['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA','Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio'],['Hair.T_Cort.ratio', 'Hair.P_Cort.ratio'],
            #                         ['Hair.T','Hair.T_Cort.ratio'],['Hair.P','Hair.P_Cort.ratio'],['Hair.DHEA','Hair.Cort_DHEA.ratio']]
            
            hormones_combination = Find_better_features.get_best_combinations(input_file)
            n_repeats = 100
            ouput_directory = "F:/Ruti/AnalysisWithPython/"
            title_file = sex + "_" + type + "_" + str(n_repeats)
            list_models = ["SVC_linear","SVC_rbf","random_forest","logistic","GaussianNB"]
            total_results_probability = pd.DataFrame()
            total_results_accuracy = pd.DataFrame()
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
            type = "hierarchy_randomization"
            sex = 'male'
            n_repeats = 100
            ouput_directory = "F:/Ruti/AnalysisWithPython/"
            select_column_prob = 0 
            title_file = sex + "_" + type + "_" + str(n_repeats)
            # Load the Pickle file
            with open(ouput_directory + title_file + '.pkl', "rb") as f:
                data = pickle.load(f)
                
            new_obj = plot_data(data, title_file, ouput_directory)
            new_obj(select_column_prob)
            a=1
        elif choice == "5":
                 type = "I_status"
                 sex = "all"
                 features = {}
                 choice_m = "3"
                 n_repeats = 100
                 hormones = ['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA','Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio']
                 output_file ='F:\Ruti\AnalysisWithPython\data_to_use.xlsx'
                 #models_list = ["SVC_linear","random_forest","logistic"] "SVC_rbf" doesnt work
                 models_list = ["SVC_linear","random_forest","logistic"]
                 ouput_directory = "F:/Ruti/AnalysisWithPython/"
                 title_file = 'Better_features' + '_' + sex + "_" + type + "_" + str(n_repeats)
                 
                 for model_name in models_list:
                    new_obj = Find_better_features(output_file)
                    final_features = new_obj(model_name,n_repeats, sex, choice_m, hormones,type)
                    features[model_name] = {'sex' : sex, 'type' : type , 'final_features' : final_features}
                    a=1
                    
                    # # Save the dictionary
                 with open(ouput_directory + title_file + '.pkl', 'wb') as f:
                    pickle.dump(features, f)
                  
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")




if __name__ == "__main__":
    main_menu()