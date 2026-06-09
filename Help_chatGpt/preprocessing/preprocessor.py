from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ============================================================
# PREPROCESSOR
# ============================================================

class BehaviorPreprocessor:
     def __init__(self):
          self.imputer = SimpleImputer(strategy="median")
          self.scaler = StandardScaler()
    
     def preprocess(self, df):

        metadata_df = df.iloc[:, :7].copy()
        # keep numeric only
        behavior_df = df.iloc[:, 7:].copy()

        behavior_df = behavior_df.select_dtypes(
            include=np.number
        )

        # impute missing values
        X = self.imputer.fit_transform(
            behavior_df
        )

        # scale
        X_scaled = self.scaler.fit_transform(X)

        return (
            behavior_df,
            X_scaled,
             metadata_df
        )

          