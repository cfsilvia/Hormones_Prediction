import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from plot_utils import compute_shap_values

#Calculate SHAP values for the full models trained on the entire dataset
def train_full_models_and_shap(hormones_arch, metadata_cols, aligned_true):
    feature_cols = [c for c in hormones_arch.columns
                    if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2']
    X_full = hormones_arch[feature_cols].select_dtypes(include=[np.number]).values

    le = LabelEncoder()
    y_full = le.fit_transform(aligned_true)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    full_models = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', max_iter=10000),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(30,), alpha=0.1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
                                  eval_metric='mlogloss', objective='multi:softmax', num_class=3),
    }

    shap_results = {}
    for mname, model in full_models.items():
        if mname == 'XGBoost':
            model.fit(X_full, y_full)
            shap_results[mname] = compute_shap_values(model, X_full, mname)
        else:
            model.fit(X_scaled, y_full)
            shap_results[mname] = compute_shap_values(model, X_scaled, mname)
    return shap_results