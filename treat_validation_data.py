import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from learning_data import learning_data


class treat_validation_data:
    def __init__(self,train_file, validation_file,columns_validation):
         self.data_train = pd.read_excel(train_file,sheet_name="All_data")
         self.data_validation = pd.read_excel(validation_file)
         self.selected_columns = columns_validation
         
         
    '''
    input: data sex, choice, hormones panel architype
    select: the data either for hierarchy or architype
    '''
    def select_data(self,data,sex = None, choice = None, hormones = None, architype = None):
        #select sex
        
        selected_data = pd.DataFrame()
        if sex == "all":
          selected_data = data
        else:
          selected_data = data[data['sex'] == sex]
          
        #select hormones and status
        if choice == "2":
            selection = hormones[:] #creates a shallow copy
            selection.append('status')
            selected_data = selected_data.loc[:,selection]
        elif choice == "3":
             selection = hormones[:]
             selection.append(architype)
             selected_data = selected_data.loc[:,selection]
        return selected_data     
         
    '''
    input: table with all data
    output: take relevant  data  and do all the combinations
    '''
    def arrange_validation(self):
        data = self.data_validation.loc[:,self.selected_columns]
        data = self.ratios_hormones(data)
        return data
    
    '''
    input : all data
    output : features and classification data
    '''
    def get_features_classification(self,data):
        X = data.iloc[:,0:(data.shape[1]-1)]
        y = data.iloc[:,(data.shape[1]-1)]
        return X,y
    
    '''
    input = train and test features data
    output = normalized train data , and test data normalize as train data
              each column is normalized independent of the others
    ''' 
    def normalization(self,X_train,X_val):
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled    
    
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
      input: selected data
      output: selected data with ratio
    '''
    @staticmethod
    def ratios_hormones(data):  
        #get the last four columns
        # last_four_columns = data.iloc[:,-4:]
        last_four_columns = data.iloc[:,2:]
        #generate all pairwis
        for i, col1 in enumerate(last_four_columns.columns):
             for j, col2 in enumerate(last_four_columns.columns):
                 if j != i:
                     ratio_name = f"{col1}_to_{col2}"
                     data[ratio_name] = last_four_columns[col1] / last_four_columns[col2]
        return data
    
    def __call__(self,model_name = None,n_repeats = None, sex = None, choice = None, hormones = None, architype = None): 
        results_dict = {}
        #arrange validation
        validation_data = self.arrange_validation()
        #arrange train data
        if choice == "2":
            selected_data_train = self.select_data(self.data_train,sex, choice, hormones)
            selected_data_validation = self.select_data(validation_data,sex, choice, hormones)
        elif choice == "3":
            selected_data_train  = self.select_data(self.data_train,sex, choice, hormones,architype)
            selected_data_validation  = self.select_data(validation_data,sex, choice, hormones,architype)
        
        #train the system with all original data and take the model (do the validation the same fitting as the train)
        #normalize the train data-use normalization for validation data
        X_train,y_train = self.get_features_classification(selected_data_train)
        X_val,y_val = self.get_features_classification(selected_data_validation)
        X_train_scaled, X_val_scaled = self.normalization(X_train,X_val)
        #balance the train data by using smote
        X_train_resampled, y_train_resampled = self.balance_data(X_train_scaled,y_train)
        #Convert the categorical data into binary
        y_train_resampled, classes = self.label_encoded(y_train_resampled)
        y_val, classes = self.label_encoded(y_val)
        classes = list(classes)
        #learn the system and get with validation the testing data
        new_obj = learning_data(X_train_resampled,X_val_scaled,y_train_resampled, 
                                y_val,model_name)

        probabilities, accuracy,y_pred, classes_num,cm,precision,recall,roc_auc,fpr,tpr,f1,mse,r2,model_params  = new_obj()
        #save into a dictionary
        results_dict['classes'] = classes
        results_dict['model'] = model_name
        results_dict['prob'] = probabilities
        results_dict['labels_pred'] = y_pred
        results_dict['true_labels'] = y_val
        results_dict['confusion_matrix'] = cm
        results_dict['accuracy'] = accuracy
        results_dict['precision'] = precision
        results_dict['recall'] = recall
        results_dict['roc_score'] = roc_auc
        results_dict['roc_fpr'] = fpr
        results_dict['roc_tpr'] = tpr
        results_dict['fscore'] = f1
        results_dict['mean_square_error'] = mse
        results_dict['r_square'] = r2
        results_dict['model_params'] = model_params
        
        
        return results_dict
        
        
        
        a=1