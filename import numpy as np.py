import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Example dataset: 30 samples belonging to 5 groups
X = np.arange(30).reshape(-1, 1)  # Features (just an index for simplicity)
y = np.random.randint(0, 2, 30)  # Labels (binary classification example)
groups = np.repeat([1, 2, 3, 4, 5,6], 5)  # 5 groups, each with 6 samples

# Define the splitter
gss = GroupShuffleSplit(n_splits=1, test_size=0.3,random_state=np.random.randint(1000))

# Perform the split
train_idx, test_idx = next(gss.split(X, y, groups))

# Split the data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]



a=1
