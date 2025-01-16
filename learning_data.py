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
from sklearn.metrics import mean_squared_error, r2_score


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
         model = SVC(kernel='rbf',probability=True)
      elif self.model_name == "random_forest":
          model = RandomForestClassifier(n_estimators = 200, random_state=42) 
          #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
      elif self.model_name == "logistic":
          model = LogisticRegression(max_iter=1000, random_state=42)
      elif self.model_name == 'decision_tree':
          model = DecisionTreeClassifier(random_state=42)
      elif self.model_name == "k_neighbors":
          model = KNeighborsClassifier(n_neighbors=5)
      elif self.model_name == "adaboost":
          model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
      elif self.model_name == "qda":
          model = QuadraticDiscriminantAnalysis()
      elif self.model_name == "GaussianNB":
           model = GaussianNB()
            
      model.fit(self.X_train, self.y_train)
      return model
  
    '''
    input: training data of boostrap data
    output: model of several............
    '''
    def train_model_b(self,X_bootstrap, y_bootstrap_permuted):
      if self.model_name == "SVC_linear":
         model = SVC(kernel='linear', probability=True,random_state=42)    
      elif self.model_name == "SVC_rbf":
         model = SVC(kernel='rbf',probability=True)
      elif self.model_name == "random_forest":
          model = RandomForestClassifier(random_state=42) 
      elif self.model_name == "logistic":
          model = LogisticRegression(max_iter=1000, random_state=42)
      elif self.model_name == 'decision_tree':
          model = DecisionTreeClassifier(random_state=42)
      elif self.model_name == "GaussianNB":
           model = GaussianNB()
            
      model.fit(X_bootstrap, y_bootstrap_permuted)
      return model
  
  
  
  
    '''
    input: test data, model
    output: probabilities and metrics
    '''
    def test_model(self,model):
        y_pred = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:\n", cm)
        # Calculate precision, recall, and F1-score-for each class
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average=None)
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        
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
        
        Aux = pd.DataFrame(self.y_test)
        Aux['binary'] = 0
        Aux.loc[Aux.iloc[:,0] == classes[0],'binary'] = 1
        labels_true = Aux['binary']
        roc_auc = roc_auc_score(labels_true, probabilities[:,0])
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels_true, probabilities[:,0])
        #
        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred) #mean squared error
        r2 = r2_score(self.y_test, y_pred) #rsquared score

        
        return probabilities, accuracy,y_pred, classes,cm,precision,recall, roc_auc,fpr,tpr,f1,mse,r2
    
    '''
    input: train data
    output : train data bootstrapped and the labels are permutated to break the relation between features and labels
     '''
    def randomize_train_data(self):
        # Bootstrap sample
        indices = np.random.choice(range(len(self.X_train)), size=len(self.X_train), replace=True)
        X_bootstrap = self.X_train[indices]
        y_bootstrap = self.y_train[indices] 
        
        # Permute labels to eliminate correlation
        y_bootstrap_permuted = np.random.permutation(y_bootstrap)
        
        return X_bootstrap, y_bootstrap_permuted
    
    
    

    def __call__(self):
        #original data
        model = self.train_model()
        probabilities, accuracy,y_pred, classes,cm,precision,recall,roc_auc,fpr,tpr, f1,mse, r2 = self.test_model(model)
        
        #for bootstrapping
        # n_bootstraps = 100
        # permuted_accuracies = []
        # for _ in range(n_bootstraps): # do several times
        #     X_bootstrap, y_bootstrap_permuted = self.randomize_train_data()       
        #     model_b = self.train_model_b(X_bootstrap, y_bootstrap_permuted )
        #     probabilities_b, accuracy_b,y_pred_b, classes_b,cm_b,precision_b,recall_b,roc_auc_b,fpr_b,tpr_b, f1_b = self.test_model(model_b)
        #     permuted_accuracies.append(accuracy_b)
        # accuracies_bootstraps = np.mean(permuted_accuracies)
            
        return probabilities, accuracy,y_pred, classes,cm,precision,recall,roc_auc,fpr,tpr,f1,mse,r2,model #,accuracies_bootstraps