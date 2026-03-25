import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


class treat_continous_labels:

    def __init__(self, data, output_dir):
        self.data = data
        self.output_dir = output_dir
        
    def __call__(self):

        X, y, feature_names = self.load_data()


        y_true, y_pred, train_errors, test_errors, models = self.loocv_pipeline(X, y)

        self.overfitting_diagnostics(train_errors, test_errors)
        self.confusion_matrix_arcs(y_true, y_pred)

        self.shap_values(models, X, feature_names)


    ################ LOAD DATA ################

    def load_data(self):

        data = pd.read_excel(self.data)

        X = data.drop(
            ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips',
             'Last.day.Glicko','Animal','Arch1','Arch2','Arch3','Arch4'], axis=1)

        y = data[['Arch1','Arch2','Arch3','Arch4']]

        feature_names = X.columns.tolist()

        return X, y, feature_names


    ################ MODEL ################

    def build_model(self):

        return XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )


    def train_arc_models(self, X_train, y_train):

        models = {}

        for arc in ['Arch1','Arch2','Arch3','Arch4']:

            model = self.build_model()
            model.fit(X_train, y_train[arc])

            models[arc] = model

        return models


    ################ PREDICTION ################

    def predict_arcs(self, models, X):

        preds = []

        for arc in ['Arch1','Arch2','Arch3','Arch4']:

            p = models[arc].predict(X)
            preds.append(p)

        preds = np.vstack(preds).T

        return preds


    def normalize_probs(self, preds):

        preds = np.clip(preds, 0, None)
        preds = preds / preds.sum(axis=1, keepdims=True)

        return preds


    ################ METRICS ################

    def compute_rmse(self, y_true, y_pred):

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        return np.sqrt(mean_squared_error(y_true, y_pred))


    ################ LOOCV ################

    def loocv_pipeline(self, X, y):

        loo = LeaveOneOut()

        y_true = []
        y_pred = []

        train_errors = []
        test_errors = []

        for train_idx, test_idx in loo.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            models = self.train_arc_models(X_train, y_train)

            preds = self.predict_arcs(models, X_test)
            preds = self.normalize_probs(preds)[0]

            y_true.append(y_test.values[0])
            y_pred.append(preds)

            train_preds = self.predict_arcs(models, X_train)
            train_preds = self.normalize_probs(train_preds)

            train_rmse = np.mean([
                self.compute_rmse(y_train.iloc[:, i], train_preds[:, i])
                for i in range(4)
            ])

            train_errors.append(train_rmse)

            test_rmse = np.mean([
                self.compute_rmse([y_test.iloc[0, i]], [preds[i]])
                for i in range(4)
            ])

            test_errors.append(test_rmse)

        return np.array(y_true), np.array(y_pred), train_errors, test_errors, models


    ################ DIAGNOSTICS ################

    def overfitting_diagnostics(self, train_errors, test_errors):

        plt.figure(figsize=(10,6))

        plt.plot(train_errors, label='Train RMSE')
        plt.plot(test_errors, label='Test RMSE')

        plt.xlabel('LOOCV Iteration')
        plt.ylabel('RMSE')

        plt.title('Train vs Test RMSE across LOOCV iterations')

        plt.legend()
        plt.show()


    ################ CONFUSION MATRIX ################

    def confusion_matrix_arcs(self, y_true, y_pred):

        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true_labels, y_pred_labels)

        plt.figure(figsize=(8,6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Arch1','Arch2','Arch3','Arch4'],
            yticklabels=['Arch1','Arch2','Arch3','Arch4']
        )

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title("Dominant Archetype Confusion Matrix")

        plt.show()


    ################ SHAP ################

    def shap_values(self, models, X, feature_names):

        print("\nComputing SHAP explanations\n")

        for arc in models:

            print("SHAP for", arc)

            explainer = shap.TreeExplainer(models[arc])
            shap_vals = explainer.shap_values(X)

            shap.summary_plot(
                shap_vals,
                X,
                feature_names=feature_names,
                plot_type="bar",
                show=True
            )