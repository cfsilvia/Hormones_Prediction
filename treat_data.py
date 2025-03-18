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

class treat_data:
    def __init__(self,output_file,select_pairs,change_status):
         self.output_file = output_file
         self.data = pd.read_excel(output_file,sheet_name="All_data")
         self.select_pairs = select_pairs
         self.change_status = change_status
        
         
    '''
    input: data sex, choice, hormones panel architype
    select: the data either for hierarchy or architype
    '''
    def select_data(self,sex = None, choice = None, hormones = None, architype = None):
        #select sex
        
        selected_data = pd.DataFrame()
        if sex == "all":
          selected_data = self.data
        else:
          selected_data = self.data[self.data['sex'] == sex]
          
        #select given data
        if self.select_pairs:
          selected_data = selected_data[selected_data['Hierarchy'].isin(self.select_pairs)]
        #assign the correct status
        if self.change_status:
           selected_data['status'] = selected_data['Hierarchy'].where(selected_data['Hierarchy'].isin(self.select_pairs))
         
        #select hormones and status
        if choice == "2":
            #add a columns to distinguish each arena experiment with numbers, beggining from 1
            selected_data = selected_data.copy()
            selected_data['groups'] = pd.factorize(selected_data['Experiment'])[0] + 1
            selection = hormones[:] #creates a shallow copy
            selection.append('status')
            selection.append('groups')
            selected_data = selected_data.loc[:,selection]
            
            
        elif choice == "3":
             selection = hormones[:]
             selection.append(architype)
             selected_data = selected_data.loc[:,selection]
        return selected_data
   
    '''
      input :data
      output: split data 70% 30%-for personalty
              consider the data of the same experiment to be together during the splitting
      '''
    def split_data(self,data,choice):
        
        
        if choice == "3":
            X = data.iloc[:,0:(data.shape[1]-1)]
            y = data.iloc[:,(data.shape[1]-1)]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                              stratify=y, shuffle=True, 
                                                              random_state=np.random.randint(20000))
        elif choice == "2":
          X = data.iloc[:,0:(data.shape[1]-2)]
          y = data.iloc[:,(data.shape[1]-2)]
          
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                stratify=y, shuffle=True, 
                                                                random_state=np.random.randint(20000))
          
          # # Define the splitter according groups
          # gss = GroupShuffleSplit(n_splits=1, test_size=0.3,random_state=np.random.randint(1000)) 
          # # Perform the split
          # train_idx, test_idx = next(gss.split(X, y, data['groups']))
          # # Split the data
          # X_train, X_test = X.iloc[train_idx.tolist()], X.iloc[test_idx.tolist()]
          # y_train, y_test = y.iloc[train_idx.tolist()], y.iloc[test_idx.tolist()]
          
          # print("Train groups:", np.unique((data['groups']).iloc[train_idx.tolist()]))
          # print("Test groups:", np.unique((data['groups']).iloc[test_idx.tolist()]))
          
          a=1
          
        return X_train, X_test, y_train, y_test
      
      
    '''
    input: data
    output : separate features from classification
    '''  
    def separate_data(self, data,choice):
        if choice == "3":
            X = data.iloc[:,0:(data.shape[1]-1)]
            y = data.iloc[:,(data.shape[1]-1)]   
        elif choice == "2":
          X = data.iloc[:,0:(data.shape[1]-2)]
          y = data.iloc[:,(data.shape[1]-2)]
          
        return X,y 
      
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
      #find K-neighbors
      classifies = np.unique(y)
      if np.sum(y == classifies[0]) < np.sum(y == classifies[1]):
        k_neighbors = np.sum(y == classifies[0]) -1
      else:
        k_neighbors = np.sum(y == classifies[1]) -1
        
      #with auto balanced exactly the 2 populations
      #smote = SMOTE(sampling_strategy='auto', k_neighbors = k_neighbors)
      smote = SMOTE(sampling_strategy='auto', k_neighbors = k_neighbors, random_state = 42) #take 4 since minority group is 5, 42 for reproducible
      #smote = SMOTE(sampling_strategy='auto', k_neighbors = k_neighbors) 
      X_resampled, y_resampled = smote.fit_resample(X, y) 
      return X_resampled, y_resampled 
    
    '''
    input: categorical variables of y
    output: convertion into binary data
    '''
    def label_encoded(self,ydata):
      yunique_values = ydata.unique()
      if self.change_status:
        custom_mapping = {self.select_pairs[0] : 1, self.select_pairs[1] : 0}
      else:
        custom_mapping = {'alpha' : 1, 'I' : 1, 'A' : 1, 'B' : 1, 'P' : 1, 'O' : 0,'submissive' : 0}
      ydata_labeled = ydata.map(custom_mapping)
      
      yunique_numeric = [custom_mapping[label] for label in yunique_values]
      combined = list(zip(yunique_values, yunique_numeric))

      # Sort based on numeric values
      sorted_combined = sorted(combined, key=lambda x: x[1])
      # Extract the sorted labels and numeric values
      sorted_labels, sorted_numeric_values = zip(*sorted_combined)
      
      
      return ydata_labeled, sorted_labels
    
    '''
    input : list of features
    output: get unique list of features together with a list of frequencies for each list, in addition
            a dictionary with a list of index for each feature
    '''
    @staticmethod
    def  find_unique_features(lists):
       unique_lists = []
       frequencies = []
       indices_dict = {} #for storing the indices for each feature
       seen = set()
       for i, lst in enumerate(lists):
         t=tuple(lst)
         if t not in seen:
           seen.add(t)
           unique_lists.append(lst)
           frequencies.append(1)
           indices_dict[t] = [i]
         else:
           frequencies[unique_lists.index(lst)] += 1
           indices_dict[t].append(i)
       return unique_lists, frequencies, indices_dict
     
    '''
     input: unique features with respective index of original list and frequency
     output: filter features with corresponding index in the original list, filter according to a frequency larger than 3 
     '''
    @staticmethod
    def filter_data(unique_lists, frequencies, indices_dict,n_repeats):
        filtered_lists = []
        filtered_frequencies = []
        filtered_indices_dict = {}
        
        for lst, freq  in zip(unique_lists, frequencies):
            t= tuple(lst)
            if freq > (n_repeats*2)/100: #keep only  features which appear more than 3 times or 2% larger
              filtered_lists.append(lst)
              filtered_frequencies.append(freq)
              filtered_indices_dict[t] = indices_dict[t]
        

        return filtered_lists,filtered_frequencies, filtered_indices_dict
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
    input: selected data, choice and selected index
    output: split data and selected data- script assure that the splitting are uniques
    '''
    def split_and_check(self,selected_data,choice,previous_splits):
        while True:
             X_train, X_test, y_train, y_test = self.split_data(selected_data,choice)
             train_indices = y_train.index.tolist()
             test_indices = y_test.index.tolist()
             
             # Convert to frozenset to allow set comparison
             split_pair = (tuple(train_indices), tuple(test_indices))
             # Check if the split already exists and has alpha
             if split_pair not in previous_splits:
               if y_test.str.replace(" ", "", regex=True).isin(['alpha']).any():
                 previous_splits.append(split_pair) #store split pair
                 break #get out from the while loop
             else:
               print("duplicate split or no alpha in test data")
               
        return X_train, X_test, y_train, y_test, previous_splits
    
    '''
    auxiliary method
    '''  
    @staticmethod
    def get_selected_data(parameter,indices):
       selected_data = [parameter[i] for i in indices]
       return selected_data
     
     
      
    '''
    input: selected data
    output : after splitting and learning get dictionary 
    '''
    def train_learning(self,selected_data,normalization, model_name=None,n_repeats = None,choice = None,findFeatureMethod = None, hormones = None):
         threshold_ratio = 0.6
         #create dictionary of the results
         results_dict = {}
         predictions,predicted_probs,true_labels, features,predictions_b,predicted_probs_b  = [], [], [], [],[],[]
         # Set up Leave-One-Out Cross-Validation (LOOCV)
         loo = LeaveOneOut()
         
         #separate features from classification
         X,y =self.separate_data(selected_data,choice)
         feature_counts = np.zeros(X.shape[1])
         
         # LOOCV Loop: For each iteration, one sample is held out as the test sample.
         for train_idx, test_idx in loo.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            #normalize the train data-use normalization for test data
            if normalization:
               X_train_scaled, X_test_scaled = self.normalization(X_train,X_test)
            else:
               X_train_scaled = X_train.to_numpy()
               X_test_scaled = X_test.to_numpy()
                
            #balance the train data by using smote
            X_train_resampled, y_train_resampled = self.balance_data(X_train_scaled,y_train)
            #make correction  if there are ratios
           # X_train_resampled = treat_data.addRatios(X_train_resampled,hormones)
            #Convert the categorical data into binary
            y_train_resampled, classes = self.label_encoded(y_train_resampled)
            classes_real = list(classes)
            y_test, classes = self.label_encoded(y_test)
            #check before
            new_obj = learning_data(X_train_resampled,X_test_scaled,y_train_resampled, y_test,model_name)
                
            y_pred_b, y_prob_b, y_test = new_obj()
                
            predictions_b.append(y_pred_b)
            predicted_probs_b.append(y_prob_b)
            
             #   

            #select features for Xtrain
            features_obj = Find_better_features(X_train_resampled,y_train_resampled,X_train)
            try:
                if findFeatureMethod == 'RFCEV':
                  selected_features, index_features = features_obj(model_name)
                elif findFeatureMethod == 'statistical':
                    selected_features, index_features = features_obj.get_best_features_from_statiscal()
                # Update feature selection frequency counts.
                feature_counts += index_features.astype(int)
                #learn the system take relevant features
                new_obj = learning_data(X_train_resampled[:,index_features],X_test_scaled[:,index_features],y_train_resampled, y_test,model_name)
                
                y_pred, y_prob, y_test = new_obj()
                
                predictions.append(y_pred)
                predicted_probs.append(y_prob)
                true_labels.append(y_test)
                
                #probabilities, accuracy,y_pred, classes,cm,precision,recall,roc_auc,fpr,tpr,f1 ,accuracies_bootstraps = new_obj()
                # probabilities_test, accuracy_test,y_pred_test, classes,cm_test,precision_test,recall_test,f1_test = new_obj()
                # a=1
                # prob.append(probabilities_test.astype(np.float32)), labels_pred.append(y_pred_test.astype(np.float32)), confusion_matrix.append(cm_test.astype(np.float32)), acc.append(accuracy_test.astype(np.float32)),prec.append(precision_test.astype(np.float32)), rec.append(recall_test.astype(np.float32)), 
                # fscore.append(f1_test.astype(np.float32)), features.append(selected_features.tolist())
              
            except Exception as e:
                print("no combination")
          
         #get better features   
         #stable_feature_indices = np.where(feature_counts / loo.get_n_splits(X, y) >= threshold_ratio)[0] 
         stable_feature_indices = np.argsort(feature_counts)[::-1]
         stable_feature_names = [X.columns[i] for i in stable_feature_indices[:10]] #only 10 features
         shap_values = []
         #encoded y 
         y, _ = self.label_encoded(y)
         #get shap values of the final model with stable features
     #    shap_values = Find_better_features.GetShapValues(X[stable_feature_names],y,model_name)
          
         # Calculate performance metrics over all LOOCV iterations.
         #before features selection
         accuracy_b, precision_b,recall_b, fscore_b,cm_b, balanced_acc_b,roc_auc_b = learning_data.metrics(predictions_b,predicted_probs_b,true_labels,model_name)     
         #after feature selection 
         accuracy, precision,recall, fscore,cm, balanced_acc,roc_auc = learning_data.metrics(predictions,predicted_probs,true_labels,model_name)     
         #Found the stable features  
         keys =['classes','features','prob','labels_pred','true_labels','confusion_matrix','accuracy','balanced_accuracy','precision','recall','fscore','roc_auc','shap_values','data_features',
                'features_before','confusion_matrix_b','accuracy_b','balanced_accuracy_b','precision_b','recall_b','fscore_b','roc_auc_b']
         values = [classes_real,  stable_feature_names, predicted_probs,  predictions,true_labels, cm, accuracy,balanced_acc, precision, recall, fscore,roc_auc, shap_values,X[stable_feature_names],
                   X.columns,cm_b,accuracy_b,balanced_acc_b, precision_b, recall_b, fscore_b,roc_auc_b] 
         results_dict = dict(zip(keys, values))   
         #get unique features
        #  unique_features, frequency_list, dict_index_per_feature = treat_data.find_unique_features(results_dict['features'])
        #  filtered_feature,filtered_frequencies, filtered_indices_dict= treat_data.filter_data(unique_features, frequency_list, dict_index_per_feature,n_repeats )
        #  #get for each feature important values

        #  new_dictionary = treat_data.get_dictionary_with_features(results_dict, filtered_feature,filtered_indices_dict,filtered_frequencies)
        #  a=1
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
     
          
          
  
    def __call__(self,model = None,normalization = None,n_repeats = None, sex = None, choice = None,findFeatureMethod = None, hormones = None, architype = None): 
         
         #add ratios if there are
         self.data = treat_data.addRatios(self.data,hormones)
         #select data to work with
         if choice == "2":
            selected_data = self.select_data(sex, choice, hormones)
         elif choice == "3":
            selected_data = self.select_data(sex, choice, hormones,architype)
            
         results_dict = self.train_learning(selected_data, normalization, model,n_repeats,choice,findFeatureMethod, hormones)
         
         return results_dict   
         # if it is required randomize the data
        #  if self.randomization:
        #    selected_data = self.randomize_data(selected_data) 
         #   
         
