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
           model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            
      model.fit(self.X_train, self.y_train)
      return model
  
   
  
    '''
    input: test data, model
    output: probabilities and metrics
    '''
    def test_model(self,model):
        
      probabilities_test, accuracy_test,y_pred_test, classes,cm_test,precision_test,recall_test,f1_test = self.calculate_metrics(self.X_test,self.y_test,model)
      #probabilities_train, accuracy_train,y_pred_train, classes,cm_train,precision_train,recall_train,f1_train = learning_data.calculate_metrics(self.X_train,self.y_train,model)
        
      return  probabilities_test, accuracy_test,y_pred_test, classes,cm_test,precision_test,recall_test,f1_test
    
    '''
    input: test and model
    output: better features with shap values
    ''' 
    def find_features(self, model):
        #compute Shap values
          explainer = shap.Explainer(model)
          shap_values = explainer(self.X_test)
          
          #agregate shape values
          if self.model_name == "xgboost":
              mean_abs_shap = np.abs(shap_values.values).mean(axis=0) #take only dominant
              shapValues = shap_values.values
          else:
              mean_abs_shap = np.abs(shap_values.values[:,:,1]).mean(axis=0) #take only dominant
              shapValues = shap_values.values[:,:,1]
              
          return shapValues, mean_abs_shap
    
    '''
    input: model name
    output: object of the model
    '''
    @staticmethod
    def model_definition(model_name):
      if model_name == "SVC_linear":
         model = SVC(kernel='linear', probability=True,random_state=42)    
      elif model_name == "SVC_rbf":
         model = SVC(kernel='rbf',C=0.1, gamma = 0.1, class_weight = 'balanced',probability=True,random_state=42)
      elif model_name == "random_forest":
          model = RandomForestClassifier(n_estimators = 50, random_state=42, n_jobs = -1) #it was 50
          #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
      elif model_name == "logistic":
          model = LogisticRegression(max_iter=1000, random_state=42,penalty ='l2',C=0.1)
      elif model_name == "decision_tree":
          model = DecisionTreeClassifier(random_state=42)
      elif model_name == "k_neighbors":
          model = KNeighborsClassifier(n_neighbors=5)
      elif model_name == "adaboost":
          model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
      elif model_name == "qda":
          model = QuadraticDiscriminantAnalysis()
      elif model_name == "GaussianNB":
           model = GaussianNB()
      elif model_name == "xgboost":
           #model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
           model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        
      return model
    
    '''
    input: test data
    output: get prob
    '''
    def simple_test_model(self,model):
      
      y_pred = model.predict(self.X_test)
      y_prob = model.predict_proba(self.X_test)
      
      return y_pred, y_prob, self.y_test
    
    '''
     input: features and classification
     output metrics
     ''' 
    @staticmethod     
    def calculate_metrics(self,X,y,model):
      custom_threshold = 0.4 #only for linear models
      y_pred = model.predict(X)
      probabilities = model.predict_proba(X)
      #only for linear models to adjust trigger for alphs
     
    #  if (self.model_name == "SVC_rbf") or (self.model_name == "SVC_linear") or (self.model_name == "logistic"):
   #   y_pred = (probabilities[:,1] >= custom_threshold).astype(int)
      
      # Compute confusion matrix
      cm = confusion_matrix(y, y_pred)
      print("Confusion Matrix:\n", cm)
        # Calculate precision, recall, and F1-score-for each class
      precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred,zero_division=0, average=None,)
      print("\nClassification Report:\n", classification_report(y, y_pred,zero_division=0))
        
      # Extract TP, TN, FP, FN
      TP = cm[1, 1]
      TN = cm[0, 0]
      FP = cm[0, 1]
      FN = cm[1, 0]

      # Calculate accuracy
      accuracy = (TP + TN) / np.sum(cm)
      classes = model.classes_
        
         # Calculate the AUC - ROC score
         #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) \
         #from prediction scores. the classifier should be binary
        
        # Aux = pd.DataFrame(self.y_test)
        # Aux['binary'] = 0
        # Aux.loc[Aux.iloc[:,0] == classes[0],'binary'] = 1
        # labels_true = Aux['binary']
        # roc_auc = roc_auc_score(labels_true, probabilities[:,0])
        # # Compute ROC curve
        # fpr, tpr, _ = roc_curve(labels_true, probabilities[:,0])
        # #
        # # Evaluate the model
        # mse = mean_squared_error(self.y_test, y_pred) #mean squared error
        # r2 = r2_score(self.y_test, y_pred) #rsquared score

        
      return probabilities, accuracy,y_pred, classes,cm,precision,recall,f1
        
    '''
     input: features and classification
     output metrics
     '''      
    def metrics(predictions,predicted_probs,true_labels,model):
      custom_threshold = 0.5 #by default it is 0.5
      #predictions = learning_data.refine_predictions(predictions, predicted_probs,custom_threshold)
    #  if (self.model_name == "SVC_rbf") or (self.model_name == "SVC_linear") or (self.model_name == "logistic"):
      #predictions = (class2_probs >= custom_threshold).astype(int)
      
      # Compute confusion matrix
      cm = confusion_matrix(true_labels, predictions)
      print("Confusion Matrix:\n", cm)
        # Calculate precision, recall, and F1-score-for each class
      precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions,zero_division=0, average=None,)
      print("\nClassification Report:\n", classification_report(true_labels, predictions,zero_division=0))
        
      # Extract TP, TN, FP, FN
      TP = cm[1, 1]
      TN = cm[0, 0]
      FP = cm[0, 1]
      FN = cm[1, 0]

      # Calculate accuracy
      accuracy = (TP + TN) / np.sum(cm)
      balanced_acc = balanced_accuracy_score(true_labels, predictions)
      class1_probs = [arr[0][1] for arr in predicted_probs]
      roc_auc = roc_auc_score(true_labels, class1_probs)
      #classes = model.classes_
      
      return accuracy, precision,recall, f1,cm, balanced_acc,roc_auc
    
    def refine_predictions(predictions, predicted_probs,custom_threshold):
      for i,arr in enumerate(predicted_probs):
        if arr[0][1] >= custom_threshold:
          predictions[i] = np.ones(1,dtype=int)
        
      return predictions
    
    def __call__(self):
        #original data
        model = self.train_model()
        #probabilities_test, accuracy_test,y_pred_test, classes,cm_test,precision_test,recall_test,f1_test= self.test_model(model)
        #shap_values, mean_abs_shap = self.find_features(model)
        y_pred, y_prob, y_test = self.simple_test_model(model)
        
        return y_pred, y_prob, y_test   
        #return probabilities_test, accuracy_test,y_pred_test, classes,cm_test,precision_test,recall_test,f1_test # shap_values, mean_abs_shap