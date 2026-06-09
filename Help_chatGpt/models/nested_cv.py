import numpy as np
from sklearn.model_selection import (RepeatedStratifiedKFold)
from sklearn.metrics import (balanced_accuracy_score,f1_score)

from models.gaussian_nb_model import GaussianNBPipeline

class NestedCrossValidator:
    def __init__(self, model):
        self.model = GaussianNBPipeline()
        self.outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
        self.all_bal_scores = []
        self.all_f1_scores = []
        self.y_true_all = []
        self.y_pred_all = []

    def run(self, X, y):
        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y)):
            print(f"\nOuter Fold {fold_idx + 1}")
            X_train = X.iloc[train_idx] 
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            #inner grid search
            self.model.fit(X_train, y_train)
            best_model = (self.model.get_best_model())
            y_pred = best_model.predict(X_test)

            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            self.all_bal_scores.append(bal_acc)
            self.all_f1_scores.append(f1)
            self.y_true_all.extend(y_test)
            self.y_pred_all.extend(y_pred)
            print(f"Balanced Accuracy: " f"{bal_acc:.3f}")
            print(f"Macro F1: " f"{f1:.3f}")
            print(self.model.get_best_params())

        #find  a model of all the data
        self.model.fit(X, y)
        final_model = self.model.get_best_model()
        print("\nFinal model trained on all data:")
        print(self.model.get_best_params())


        return {
            "best_model": final_model,


            "bal_scores": np.array(
                self.all_bal_scores
            ),

            "f1_scores": np.array(
                self.all_f1_scores
            ),

            "y_true": np.array(
                self.y_true_all
            ),

            "y_pred": np.array(
                self.y_pred_all
            )
        }
