import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import  LeaveOneOut, StratifiedKFold, train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from learning_data import learning_data
from sklearn.model_selection import GroupShuffleSplit
from Find_better_features import Find_better_features
from collections import defaultdict
from sklearn.preprocessing import RobustScaler
from collections import Counter

class treat_data:
    def __init__(self,input_file,select_pairs):
         self.input_file = input_file
         self.data = pd.read_excel(input_file,sheet_name="All_data")
         self.select_pairs = select_pairs
        
        
         
    '''
    input: data sex
    select: get X features with a feature that describe the status, y for classification information data and labels
    '''
    def select_data(self,sex = None, hormones = None):
        #select sex
        
        selected_data = pd.DataFrame()
        if sex == "all":
          selected_data = self.data
         # selected_data['sex_num'] = selected_data['sex'].map({'female':0, 'male' : 1})
        else:
          selected_data = self.data[self.data['sex'] == sex]
          
        #select hierarchy mice
        if self.select_pairs:
          selected_data = selected_data[selected_data['Hierarchy'].isin(self.select_pairs)]
       
        #select the columns you are intrested
        selected_data = selected_data.copy()
        selection = hormones[:] #creates a shallow copy
       # selection.append('sex_num')
        information_data = selected_data.iloc[:,[0,1,2,3,4,5,6]]
        X = selected_data.loc[:,selection] # only features only classification
        y, labels = self.label_encoded(selected_data.loc[:,'Hierarchy']) #to numbers

        return X , y , information_data, labels
    
    '''
    input: categorical variables of y
    output: convertion into number data and get sorted labels
    '''
    def label_encoded(self,ydata):
      y_unique_values = ydata.unique()
      if len(y_unique_values) == 3:
         custom_mapping = {'alpha' : 2, 'beta' : 1, 'delta' : 1,'epsilon' : 0 }
      elif len(y_unique_values) == 5:
         custom_mapping = {'alpha' : 4, 'beta' : 3, 'gamma' : 2, 'delta' : 1, 'epsilon' : 0 }
      elif len(y_unique_values) == 2:
         custom_mapping = {'alpha' : 1, 'beta' : 0, 'epsilon' : 0 , 'gamma' : 0, 'delta' : 0}
         
      ydata_num = ydata.map(custom_mapping)
      
      yunique_numeric = [custom_mapping[label] for label in y_unique_values]
      combined = list(zip(y_unique_values, yunique_numeric))

      # Sort based on numeric values
      sorted_combined = sorted(combined, key=lambda x: x[1])
      # Extract the sorted labels and numeric values
      sorted_labels, sorted_numeric_values = zip(*sorted_combined)

      return ydata_num, sorted_labels
   
    '''
    input = train and test features data
    output = normalized train data , and test data normalize as train data
              each column is normalized independent of the others
    ''' 
    def normalization(self,X_train,X_test):
        scaler = RobustScaler()  #change to   robust  for the case of ouliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
  
    '''
    input: hormones with classification
    output: add data to minority group for balance
         not we change k_neighbors
    '''
    def balance_data(self,X,y):
      # Step 1: Find the max class count
      class_counts = Counter(y)
      max_class_size = max(class_counts.values())
      # Step 2: Create a sampling_strategy dict for SMOTE
      # We want to upsample all smaller classes to the max
      sampling_strategy = {
         cls: max_class_size for cls, count in class_counts.items() if count < max_class_size
        } 
      # Step 3: Apply SMOTE with custom strategy
      smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
      X_resampled, y_resampled = smote.fit_resample(X, y)
      
      return X_resampled, y_resampled 
    
    
    
   
    
    '''
    input: dictionary
    output: new dictionary  with key the selected feature , list ofthe parameters
    '''
    @staticmethod  
    def get_dictionary_with_features(data_dict, features,indices_dict,frequencies):
      results =  defaultdict(dict)
      #sort features according to frequencies in descending order
      sorted_pairs = sorted(zip(frequencies,features), reverse = True)
      frequencies, features = zip(*sorted_pairs)
      for i,lst in enumerate(features):
        t = tuple(lst)
        results[t]['classes'] = data_dict['classes']
        results[t]['repetition_feature'] = frequencies[i]
        results[t]['prob'] = treat_data.get_selected_data(data_dict['prob'],indices_dict[t])
        results[t]['labels_pred'] = treat_data.get_selected_data(data_dict['labels_pred'],indices_dict[t])
        results[t]['confusion_matrix'] = treat_data.get_selected_data(data_dict['confusion_matrix'],indices_dict[t])
        results[t]['accuracy'] = treat_data.get_selected_data(data_dict['accuracy'],indices_dict[t])
        results[t]['precision'] = treat_data.get_selected_data(data_dict['precision'],indices_dict[t])
        results[t]['recall'] = treat_data.get_selected_data(data_dict['recall'],indices_dict[t])
        results[t]['fscore'] = treat_data.get_selected_data(data_dict['fscore'],indices_dict[t])
        
      return results

    '''
    input: selected data
    output : after splitting and learning get dictionary 
    '''
    def train_learning(self,X, y, sorted_labels,normalization, model_name=None, hormones = None,information_data = None):
         
         #create dictionary of the results
         results_dict = {}
         predictions,predicted_probs,true_labels, all_shap_values, all_sex_values, all_mice_information, all_interaction_values  = [], [], [], [], [], [], []
         number_labels = len(sorted_labels)
         # Set up Leave-One-Out Cross-Validation (LOOCV)
         loo = LeaveOneOut()
         
         # LOOCV Loop: For each iteration, one sample is held out as the test sample.
         for train_idx, test_idx in loo.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if information_data is not None:
                information_mice = information_data.iloc[test_idx]
            # #normalize the train data-use normalization for test data
            if normalization:
                X_train_scaled, X_test_scaled = self.normalization(X_train,X_test)
            else:
                X_train_scaled = X_train.to_numpy()
                X_test_scaled = X_test.to_numpy()
                
            #balance the train data by using smote  with the larger class
            X_train_resampled, y_train_resampled = self.balance_data(X_train_scaled,y_train)
           
            # #check before
            new_obj = learning_data(X_train_resampled,X_test_scaled,y_train_resampled, y_test,model_name,number_labels)
                
            y_pred, y_prob, y_test, shap_values, interaction_values = new_obj()
                
            predictions.append(int(y_pred[0]))
            predicted_probs.append(y_prob[0].tolist())
            true_labels.append(int(y_test.iloc[0]))
            all_shap_values.append(shap_values)
            all_interaction_values.append(interaction_values)
            
            if information_data is not None:
               all_mice_information.append(information_mice)
  
    #      #get shap values of the final model with stable features
    #  #    shap_values = Find_better_features.GetShapValues(X[stable_feature_names],y,model_name)

         accuracy, precision,recall, f1,cm, balanced_acc = learning_data.metrics(predictions,predicted_probs,true_labels)     
         #Found the stable features  
         keys =['classes','features','prob','labels_pred','true_labels','confusion_matrix','accuracy','balanced_accuracy','precision','recall','fscore','shap_values','interaction_shap','data_features','mice_information']
         values = [sorted_labels, X.columns, predicted_probs,  predictions,true_labels, cm, accuracy,balanced_acc, precision, recall, f1, all_shap_values,all_interaction_values,X, all_mice_information]                 
         results_dict = dict(zip(keys, values))   
    
         return results_dict
    '''
    input : data
    output: add ratios if there are
    '''
      
    @staticmethod
    def addRatios(data, hormones):
        result = [item for item in hormones if '_' in item]
        if not result:
          print("The list is empty")
        else:
         for r in result:
          parts = r.split('_')
          first_part = parts[0]
          last_part = parts[-1]
          data[r] = data[first_part]/data[last_part]
        return data
     
     
    '''
     input_data: shuffled data
     output_data: Fscore for each shuffle
     '''
    def  add_shuffling(self,selected_data,normalization, model,n_repeats,choice,findFeatureMethod, hormones):
           n_iterations = 1000
           permutations = set()
           # Create a generator with a fixed seed assure different permutation each time
           rng = np.random.default_rng(42)
           all_fscore_class1 = []
           all_fscore_class2 = []
           
           for i in range(n_iterations):
                df_permutated = selected_data.copy()
                 #permutate only the status
                df_permutated['status'] = rng.permutation(df_permutated['status'])
                results_dict = self.train_learning(df_permutated, normalization, model,n_repeats,choice,findFeatureMethod, hormones)
                fscore = results_dict['fscore']
                all_fscore_class1.append(fscore[0])
                all_fscore_class2.append(fscore[1])
                

           return all_fscore_class1, all_fscore_class2
                
                

  
    def __call__(self,model = None,normalization = None,n_repeats = None, sex = None, hormones = None): 
         
         #add ratios if there are
         self.data = treat_data.addRatios(self.data,hormones)
         
        #  #select data to work with
         X , y , information_data, sorted_labels = self.select_data(sex, hormones)
         results_dict = self.train_learning(X, y, sorted_labels, normalization, model, hormones, information_data)
        #  list_fscore = results_dict['fscore']
        #  #add shuffling 
        #  if (list_fscore[0] >= 0.6) and (list_fscore[1] >= 0.6):
        #      all_fscore_class1, all_fscore_class2 = self.add_shuffling(selected_data,normalization, model,n_repeats,choice,findFeatureMethod, hormones)
        #      results_dict['shuffle_fscore_class1'] = all_fscore_class1
        #      results_dict['shuffle_fscore_class2'] = all_fscore_class2
        #  else:
        #      results_dict['shuffle_fscore_class1'] = []
        #      results_dict['shuffle_fscore_class2'] = []
          
         
         
        #  return results_dict   
         # if it is required randomize the data
        #  if self.randomization:
        #    selected_data = self.randomize_data(selected_data) 
         #   
         return results_dict
