import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (GridSearchCV, StratifiedKFold)

class GaussianNBPipeline:

    def __init__(self):

        self.pipeline = Pipeline([(
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "clf",
                GaussianNB()
            )
        ])
        
        self.param_grid = {"clf__var_smoothing": np.logspace(-12, -6, 20)}

        self.inner_cv = StratifiedKFold(

            n_splits=4,

            shuffle=True,

            random_state=42
        )

        self.grid = GridSearchCV(

            estimator=self.pipeline,

            param_grid=self.param_grid,

            scoring="f1_macro",

            cv=self.inner_cv,

            n_jobs=-1
        )
    
    def fit(self, X, y):
      self.grid.fit(X, y)

    def predict(self, X):
      return self.grid.best_estimator_.predict(X)
    
    def get_best_model(self):

        return self.grid.best_estimator_
    
    def get_best_params(self):
          return self.grid.best_params_