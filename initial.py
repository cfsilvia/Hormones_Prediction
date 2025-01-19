from manage_data import manage_data
from manage_data_AviCondition import manage_data_AviCondition
from treat_data import treat_data
import pandas as pd
import pickle
from plot_data import plot_data
from treat_validation_data import treat_validation_data
from Find_better_features import Find_better_features
from treat_random_data import treat_random_data
#from sklearn.ensemble import AdaBoostClassifier

def main_menu():
     while True:
        print("\nMain Menu")
        print("1. Option 1: First step, create an excel with the data to use")
        print("2. Option 2: Predict hierarchy")
        print("3. Option 3: Predict personality")
        print("4. Option 4: Get graphs and tables")
        print("5. Option 5: Find better features for each model")
        print("6. Option 6: Evaluate validation data")
        print("7. Option 7: break")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == "1":
            pareto_file = 'F:/Ruti/Pareto/10.09.24_stage2_stage2_Pareto_Information.xlsx'
          #  data_file = 'F:/Ruti/PCA/PCA_PCA__10.9.24_stage2_NEW_stage2.xlsx'
            data_file = 'F:/Ruti/PCA/Copy of PCA_PCA__10.9.24_stage2_NEW_stage2_ECBs.xlsx'
            output_file ='F:\Ruti\AnalysisWithPython\data_to_use.xlsx'
            distance = 0.5
            
            new_obj = manage_data(pareto_file, data_file, output_file, distance)
           # new_obj = manage_data_AviCondition(pareto_file, data_file, output_file, distance)
            new_obj()
            
        elif choice == "2":
            type = "hierarchy_all_ratios"
            output_file ='F:\Ruti\AnalysisWithPython\data_to_use_complete.xlsx'
            sex = 'male' #female or all
            input_file = 'F:/Ruti/AnalysisWithPython/Better_features_male_hierarchy_all_ratios_400.pkl'
    #         
            hormones_combination = Find_better_features.get_best_combinations(input_file)

            n_repeats = 400
            ouput_directory = "F:/Ruti/AnalysisWithPython/"
            randomization = False # in the case want to randomize the classification
            num_permutations = 3
            title_file = sex + "_" + type + "_" + str(n_repeats)
           #for randomization
            #title_file = sex + "_" + type + "_" + str(n_repeats) + "_" + str(num_permutations)
            #list_models = ["SVC_linear","SVC_rbf","random_forest","logistic"]
            list_models = ["SVC_linear","SVC_rbf","random_forest","logistic", "decision_tree","k_neighbors","qda"]
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
            sex = 'female' #female or all
            input_file = 'F:\Ruti\AnalysisWithPython\Better_features\Better_features_male_hierarchy_binary_100.pkl'
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
            type = "hierarchy_all_ratios_hormones"
            sex = 'male'
            n_repeats = 400
            ouput_directory = "F:/Ruti/AnalysisWithPython/"
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

            
            
            a=1
        elif choice == "5":
                 type = "hierarchy_all_ratios"
                 sex = "female"
                 features = {}
                 choice_m = "2"
                 n_repeats = 400
                 hormones =['Hair.P','Hair.T',	'Hair.Cort', 'Hair.DHEA', 'Hair.P_to_Hair.T',	'Hair.P_to_Hair.Cort', 'Hair.P_to_Hair.DHEA',
                            'Hair.T_to_Hair.P', 'Hair.T_to_Hair.Cort', 'Hair.T_to_Hair.DHEA',	'Hair.Cort_to_Hair.P', 'Hair.Cort_to_Hair.T',	
                            'Hair.Cort_to_Hair.DHEA', 'Hair.DHEA_to_Hair.P', 'Hair.DHEA_to_Hair.T', 'Hair.DHEA_to_Hair.Cort']

                # hormones = ['Hair.P', 'Hair.T','Hair.Cort', 'Hair.DHEA','Hair.T_Cort.ratio', 'Hair.P_Cort.ratio','Hair.Cort_DHEA.ratio']
                 output_file ='F:\Ruti\AnalysisWithPython\data_to_use_complete.xlsx'
                 #models_list = ["SVC_linear","random_forest","logistic"] "SVC_rbf" doesnt work
                 #models_list = ["SVC_linear","random_forest","logistic", "decision_tree","k_neighbors","adaboost","qda"]
                 models_list = ["SVC_linear","random_forest","logistic", "decision_tree"]
                 ouput_directory = "F:/Ruti/AnalysisWithPython/"
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
                    
        elif choice == "6": #for validation
            type = "hierarchy_all_ratios_hormones"
            train_file ='F:/Ruti/AnalysisWithPython/data_all_hormones_vs2.xlsx'
            validation_file = 'F:/Ruti/fresh_Data/Hormones_Mice_ctrl_casp3_2F_1M_vs1.xlsx'
            sex = 'male' #female or all
            choice = "2"
            n_repeats = 1
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
            a=1       
        elif choice == "7":
            
            
            break
        else:
            print("Invalid choice. Please try again.")




if __name__ == "__main__":
    main_menu()