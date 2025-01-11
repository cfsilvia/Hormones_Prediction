import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from learning_data import learning_data

class treat_data:
    def __init__(self,output_file):
         self.output_file = output_file
         self.data = pd.read_excel(output_file,sheet_name="All_data")
        
         
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
      input :data
      output: split data 70% 30%-regular
      '''
    def split_data(self,data):
          X = data.iloc[:,0:(data.shape[1]-1)]
          y = data.iloc[:,(data.shape[1]-1)]
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                              stratify=y, shuffle=True, 
                                                              random_state=np.random.randint(1000))
          return X_train, X_test, y_train, y_test
    '''
    input = train and test features data
    output = normalized train data , and test data normalize as train data
              each column is normalized independent of the others
    ''' 
    def normalization(self,X_train,X_test):
        
        scaler = StandardScaler()
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
    
    # '''
    # input: original data
    # output: after randomization
    # '''
    # def randomize_data(self,data):
        
    #     # Bootstrap sample
    #     indices = np.random.choice(range(data.shape[0]), size=data.shape[0], replace=False)
    #     data_bootstrap = data.iloc[indices,:]
    #     data_bootstrap_permutated = data_bootstrap.copy()
        
    #     # Permute labels to eliminate correlation
    #     labels_permutated = np.random.permutation(data_bootstrap.iloc[:,data_bootstrap.shape[1]-1])
    #     data_bootstrap_permutated.iloc[:,data_bootstrap.shape[1]-1] = labels_permutated
        
    #     return data_bootstrap_permutated
    
    '''
    input: selected data
    output : after splitting and learning get dictionary 
    '''
    def train_learning(self,selected_data, model=None,n_repeats = None):
         #create dictionary of the results
         results_dict = {}
         prob = []
         labels_pred = []
         confusion_matrix = []
         acc = []
         prec = []
         rec = []
         roc_auc1 = []
         fpr1 = []
         tpr1 = []
         fscore = []
         mean_s_error = []
         r_square = []
         accuracy_boot_perm = []
         for count in range(n_repeats):
            #get train  and test data
            X_train, X_test, y_train, y_test =self.split_data(selected_data)
            #normalize the train data-use normalization for test data
            X_train_scaled, X_test_scaled = self.normalization(X_train,X_test)
            #balance the train data by using smote
            X_train_resampled, y_train_resampled = self.balance_data(X_train_scaled,y_train)
            #Convert the categorical data into binary
            y_train_resampled, classes = self.label_encoded(y_train_resampled)
            y_test, classes = self.label_encoded(y_test)
            #learn the system
            new_obj = learning_data(X_train_resampled,X_test_scaled,y_train_resampled, y_test,model)
            #probabilities, accuracy,y_pred, classes,cm,precision,recall,roc_auc,fpr,tpr,f1 ,accuracies_bootstraps = new_obj()
            probabilities, accuracy,y_pred, classes,cm,precision,recall,roc_auc,fpr,tpr,f1,mse,r2  = new_obj()
            prob.append(probabilities)
            labels_pred.append(y_pred)
            confusion_matrix.append(cm)
            acc.append(accuracy)
            prec.append(precision)
            rec.append(recall)
            roc_auc1.append(roc_auc)
            fpr1.append(fpr)
            tpr1.append(tpr)
            fscore.append(f1)
            mean_s_error.append(mse)
            r_square.append(r2)
           # accuracy_boot_perm.append(accuracies_bootstraps)
            
         results_dict['classes'] = classes
         results_dict['model'] = model
         results_dict['prob'] = prob
         results_dict['labels_pred'] = labels_pred
         results_dict['confusion_matrix'] = confusion_matrix
         results_dict['accuracy'] = acc
         results_dict['precision'] = prec
         results_dict['recall'] = rec
         results_dict['roc_score'] = roc_auc1
         results_dict['roc_fpr'] = fpr1
         results_dict['roc_tpr'] = tpr1
         results_dict['fscore'] = fscore
         results_dict['mean_square_error'] = mean_s_error
         results_dict['r_square'] = r_square
         #results_dict['accuracy_boot_perm'] = accuracy_boot_perm
          

         return results_dict
    
    
  
    def __call__(self,model = None,n_repeats = None, sex = None, choice = None, hormones = None, architype = None): 
        
         
         #select data to work with
         if choice == "2":
            selected_data = self.select_data(sex, choice, hormones)
         elif choice == "3":
            selected_data = self.select_data(sex, choice, hormones,architype)
            
         results_dict = self.train_learning(selected_data, model,n_repeats)
         
         return results_dict   
         # if it is required randomize the data
        #  if self.randomization:
        #    selected_data = self.randomize_data(selected_data) 
         #   
         
