import shap
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train a model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Explain the model
explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_test)

# Check shape (should be (n_samples, n_features, n_features))
print(np.array(shap_interaction_values).shape)

# Take interaction matrix for the first sample
interaction_matrix = shap_interaction_values[0]

# Convert to DataFrame for easier reading
interaction_df = pd.DataFrame(interaction_matrix, 
                              index=X.columns, 
                              columns=X.columns)

print(interaction_df)