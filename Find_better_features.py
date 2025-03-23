
import pandas as pd
import numpy as np
from learning_data import learning_data
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import make_scorer, f1_score
from collections import Counter
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.model_selection import KFold
from scipy.stats import mannwhitneyu
import shap

# Example: Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Find_better_features:
    def __init__(self,X_train, y_train,X_train_original):
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_with_columns = X_train_original
       
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
    
    '''
    input: train data
    output: features and index according shapley values
    '''
    def get_best_features_from_shapley_values(self,model_name):
        feature_importance = np.zeros(self.X_train.shape[1])
        shap_values_all = []
        #set up for cross validation
        n_splits = 5
        kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
        for train_idx, val_idx in kf.split(self.X_train, self.y_train): 
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            #train model
            new_obj = learning_data(X_train_fold,X_val_fold,y_train_fold, y_val_fold,model_name)
            model = new_obj.train_model()
            #compute Shap values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_fold)[:,:,1] #SHAP values for class 1 calculation on the validation
            #agregate shape values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance += mean_abs_shap
            shap_values_all.append(mean_abs_shap)
            
        sorted_idx = np.argsort(feature_importance)[::-1]
        cumulative_importance = np.cumsum(feature_importance[sorted_idx])/np.sum(feature_importance)
        #choose features covering 90% of importance
        num_features = np.argmax(cumulative_importance >= 0.90) + 1
        selected_features_indexes = sorted_idx[:num_features]
        selected_features = self.X_train_with_columns.columns[selected_features_indexes]
        
        return selected_features, selected_features_indexes
     
        
        
        
        
    '''
    input: train data
    output: features which were obtained from statistics
    '''          
    def get_best_features_from_statiscal(self):
        num_folds = 5
        kf = KFold(n_splits=num_folds)  
        p_val_features_fold = np.zeros(shape=(num_folds,self.X_train.shape[1]))
        n_fold = 0
        threshold_pval = 0.05 #or 0.1
        # Iterating over the splits
        for train_index, _ in kf.split(self.X_train):
            X_train_fold = self.X_train[train_index,:]
            y_train_fold = self.y_train[train_index]
            x_0 = X_train_fold[y_train_fold == 0]
            x_1 = X_train_fold[y_train_fold == 1]
            p_val = mannwhitneyu(x_0, x_1)[1]
            p_val_features_fold[n_fold,:] = p_val
            n_fold +=1
        feature_selected_k_fold = np.average(p_val_features_fold, axis=0) < threshold_pval
        indexes_selected_features = np.where(feature_selected_k_fold)[0]
        selected_features = self.X_train_with_columns.columns[indexes_selected_features]
        print(selected_features)
        return selected_features, feature_selected_k_fold
         
    def __call__(self,model_name = None):       
       
        # Step 4: Apply RFECV on the training data
            try:
                #get model object
                model = learning_data.model_definition(model_name)
                # Specify 'I' as the positive class
                #for rfcev
                # f1_scorer = make_scorer(f1_score, pos_label= 1)
                # #do ten folds splitting
                # rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5,shuffle=True,random_state = 42), scoring=f1_scorer, n_jobs = -1)
                # #rfecv = RFECV(estimator=model, step=1, cv=30, scoring=f1_scorer, min_features_to_select=1, n_jobs = -1)
                # rfecv.fit(self.X_train, self.y_train)
                # selected_features = self.X_train_with_columns.columns[rfecv.support_]
                
                #for rfe
                rfe = RFE(estimator=model, n_features_to_select=10, step=1)
                rfe.fit(self.X_train, self.y_train)
                selected_features = self.X_train_with_columns.columns[rfe.support_]
                
                print("Optimal number of features:", rfe.n_features_)
                print("Selected features:", selected_features)
            except Exception as e:
                print(f"An error ocurred: {e}")
        
            return selected_features,rfe.support_
    '''
    input: original data
    output: shap values
    '''
    @staticmethod
    def GetShapValues(X,y,model_name):
        final_model = learning_data.model_definition(model_name)
        final_model.fit(X,y)
        
        if (model_name == "random_forest") or (model_name == "decision_tree"):
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X)[:,:,1]
            #agregate shape values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            explainer = shap.Explainer(final_model, X)
            shap_values = explainer(X)
            shap_values = shap_values.data
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        return mean_abs_shap