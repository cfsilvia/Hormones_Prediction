import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from plot_utils import compute_shap_values

#Calculate SHAP values for the full models trained on the entire dataset
def train_full_models_and_shap(hormones_arch, metadata_cols, aligned_true, model_names=None):
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
        'RandomForest': RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
                                  eval_metric='mlogloss', objective='multi:softmax', num_class=3),
    }
    if model_names is not None:
        unknown = [name for name in model_names if name not in full_models]
        if unknown:
            raise ValueError(f'Unknown model names: {unknown}')
        full_models = {name: full_models[name] for name in model_names}

    shap_results = {}
    for mname, model in full_models.items():
        if mname in ('RandomForest', 'ExtraTrees', 'HistGB', 'XGBoost'):
            model.fit(X_full, y_full)
            shap_results[mname] = compute_shap_values(model, X_full, mname)
        else:
            model.fit(X_scaled, y_full)
            shap_results[mname] = compute_shap_values(model, X_scaled, mname)
    return shap_results
