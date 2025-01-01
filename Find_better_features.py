import pandas as pd
from treat_data import treat_data
from learning_data import learning_data
from sklearn.feature_selection import RFECV
from collections import Counter
import pickle

class Find_better_features:
    def __init__(self,output_file):
         self.output_file = output_file
         self.data = pd.read_excel(output_file,sheet_name="All_data")
    '''
    input: list of features
    output: order features 
    '''    
    @staticmethod
    def order_features(list_features):
        # Convert inner lists to tuples to make them hashable
        hashable_list = [tuple(sublist) for sublist in list_features]
        # Count occurrences
        element_counts = Counter(hashable_list)
       
        # Extract unique elements and counts
        unique_elements, counts = zip(*element_counts.items())
        
        # Sort by counts in descending order
        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True) #lambda takes counts to do the sorting
        sorted_unique_elements = [list(unique_elements[i]) for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        # Filter elements with counts greater than 4
        final_features = [sorted_unique_elements[i] for i in range(len(sorted_counts)) if sorted_counts[i] > 4]
        
        return final_features
    '''
    input : dictionary with all the results of all models
    output: best features
    '''
    @staticmethod
    def get_best_combinations(input_file):
        list_features = list()
        seen = set()
        #load pkl dictionary
        with open(input_file , "rb") as f:
                data = pickle.load(f)
        for key in data:
            for inner_list in data[key]['final_features']:
              inner_tuple1 = tuple(sorted(inner_list))
              if inner_tuple1 not in seen:
               list_features.append(inner_list)
               seen.add(inner_tuple1)
               
        return list_features
            
        
       
         
    def __call__(self,model_name = None,n_repeats = None, sex = None, choice = None, hormones = None, type = None,architype = None):       
         
        #select data to work with
        new_obj = treat_data(self.output_file)
        if choice == "2":
            selected_data = new_obj.select_data(sex, choice, hormones)
        elif choice == "3":
            architype = type
            selected_data = new_obj.select_data(sex, choice, hormones,architype)
        list_features = list()
        
        for count in range(n_repeats):
            #get train  and test data
            X_train, X_test, y_train, y_test = new_obj.split_data(selected_data)
            #normalize the train data-use normalization for test data
            X_train_scaled, X_test_scaled = new_obj.normalization(X_train,X_test)
            #balance the train data by using smote
            X_train_resampled, y_train_resampled = new_obj.balance_data(X_train_scaled,y_train)
            #find the model
            new_obj_learn = learning_data(X_train_resampled, X_test_scaled, y_train_resampled,y_test,model_name)
            model = new_obj_learn.train_model()
            
            # Step 4: Apply RFECV on the training data
            try:
                rfecv = RFECV(estimator=model, step=1, cv=5, scoring='f1')
                rfecv.fit(X_train_resampled, y_train_resampled)
                selected_features = X_train.columns[rfecv.support_]
                # Results append to list
                list_features.append(selected_features.tolist())
                
                print("Optimal number of features:", rfecv.n_features_)
                print("Selected features:", selected_features)
            except Exception as e:
                print(f"An error ocurred: {e}")
        final_features = Find_better_features.order_features(list_features)    

        return final_features