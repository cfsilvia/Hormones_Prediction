from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, r2_score,balanced_accuracy_score
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score




class learning_data:
    def __init__(self,X_train,X_test,y_train, y_test,model):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model
        
    '''
    input: training data
    output: model of several............
    '''
    def train_model(self):
      if self.model_name == "SVC_linear":
         model = SVC(kernel='linear', probability=True,random_state=42)    
      elif self.model_name == "SVC_rbf":
         model = SVC(kernel='rbf',C=0.1, gamma = 0.1, class_weight = 'balanced',probability=True,random_state=42)
      elif self.model_name == "random_forest":
          model = RandomForestClassifier(n_estimators = 50, random_state=42, n_jobs = -1) #it was 50
          #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
      elif self.model_name == "logistic":
          model = LogisticRegression(max_iter=1000, random_state=42,penalty ='l2',C=0.1)
      elif self.model_name == "decision_tree":
          model = DecisionTreeClassifier(random_state=42)
      elif self.model_name == "k_neighbors":
          model = KNeighborsClassifier(n_neighbors=5)
      elif self.model_name == "adaboost":
          model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
      elif self.model_name == "qda":
          model = QuadraticDiscriminantAnalysis()
      elif self.model_name == "GaussianNB":
           model = GaussianNB()
      elif self.model_name == "xgboost":
           #model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
          # model = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric="logloss", random_state=42)
            model = xgb.XGBClassifier(objective="multi:softprob",num_class =3,
            n_estimators = 25, max_depth =2, reg_alpha = 1.0, use_label_encoder=False, eval_metric="logloss", random_state=42)
        #   model = xgb.XGBClassifier(objective="multi:softprob", use_label_encoder=False, eval_metric="logloss", random_state=42)
      
      model.fit(self.X_train, self.y_train)
      return model
  
   

    
    '''
    input: test and model
    output: better features with shap values
    ''' 
    def find_features(self, model):
        
          #For 3 classes, shap_values will be a list with 3 arrays (one per class)
          #agregate shape values
          if (self.model_name == "xgboost") or (self.model_name == "random_forest") or (self.model_name == "decision_tree"):
              explainer = shap.TreeExplainer(model)
              shap_values = explainer.shap_values(self.X_test)   
          else:
              #compute Shap values
              explainer = shap.Explainer(model)
              shap_values = explainer(self.X_test)
              mean_abs_shap = np.abs(shap_values.values[:,:,1]).mean(axis=0) #take only dominant
              shapValues = shap_values.values[:,:,1]
              
          return shap_values
    
 
    
    '''
    input: test data
    output: get prob
    '''
    def simple_test_model(self,model):
      
      y_pred = model.predict(self.X_test)
      y_prob = model.predict_proba(self.X_test) #for the three classes
      
      return y_pred, y_prob, self.y_test
    
    
  
        
    '''
     input: features and classification
     output metrics
     '''      
    def metrics(predictions,predicted_probs,true_labels):
   
      # Compute confusion matrix
      cm = confusion_matrix(true_labels, predictions)
      print("Confusion Matrix:\n", cm)
        # Calculate precision, recall, and F1-score-for each class
      precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions,zero_division=0, average=None,)
      print("\nClassification Report:\n", classification_report(true_labels, predictions,zero_division=0))
      
      
      # Calculate accuracy
      accuracy = accuracy_score(true_labels, predictions)
      balanced_acc = balanced_accuracy_score(true_labels, predictions)
      #class1_probs = [arr[0][1] for arr in predicted_probs]
      #roc_auc = roc_auc_score(true_labels, class1_probs)
      #classes = model.classes_
      
      return accuracy, precision,recall, f1,cm, balanced_acc
    
    def refine_predictions(predictions, predicted_probs,custom_threshold):
      for i,arr in enumerate(predicted_probs):
        if arr[0][1] >= custom_threshold:
          predictions[i] = np.ones(1,dtype=int)
        
      return predictions
    
    def __call__(self):
        #original data
        model = self.train_model()
        y_pred, y_prob, y_test = self.simple_test_model(model)
        shap_values = self.find_features(model)
        
        return y_pred, y_prob, y_test , shap_values 
        