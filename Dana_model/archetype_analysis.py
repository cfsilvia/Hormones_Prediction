import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from py_pcha import PCHA


def _normalize_column_names(df):
    df = df.copy()
    df.columns = [c.replace(' ', '.') for c in df.columns]
    return df


def assign_to_nearest_vertex(pca_coords, archetypes):
    dists = np.linalg.norm(pca_coords[:, np.newaxis, :] - archetypes[np.newaxis, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    counts = [int(np.sum(assignments == v)) for v in range(archetypes.shape[0])]
    return counts


def compute_pca(data_df):
    from scipy.stats import zscore
    data_df = _normalize_column_names(data_df)
    exclude_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']
    behavior_cols = [c for c in data_df.columns if c not in exclude_cols]
    behavior_data = data_df[behavior_cols].select_dtypes(include=[np.number]).values
    behavior_data = zscore(behavior_data, nan_policy='omit')
    behavior_data = np.nan_to_num(behavior_data)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(behavior_data)
    return pca, pca_coords, behavior_cols


def apply_pca_transform(pca, data_df, behavior_cols):
    from scipy.stats import zscore
    data_df = _normalize_column_names(data_df)
    available_cols = [c for c in behavior_cols if c in data_df.columns]
    if len(available_cols) < 2:
        raise ValueError(f'Only {len(available_cols)} behavior columns found in new data')
    behavior_data = data_df[available_cols].select_dtypes(include=[np.number]).values
    behavior_data = zscore(behavior_data, nan_policy='omit')
    behavior_data = np.nan_to_num(behavior_data)
    return pca.transform(behavior_data)


def sample_and_fit_archetypes(pca_coords_all, sample_frac=0.8):
    n = pca_coords_all.shape[0]
    n_sample = max(int(n * sample_frac), 3)
    sample_indices = np.random.choice(n, n_sample, replace=False)
    pca_coords = pca_coords_all[sample_indices]

    X_for_pcha = pca_coords.T
    XC, S, C, SSE, varexlp = PCHA(X_for_pcha, 3)
    archetypes = np.array(XC.T)

    counts = assign_to_nearest_vertex(pca_coords_all, archetypes)

    return pca_coords, archetypes, varexlp, counts


def compute_archetype_probabilities(pca_coords, archetypes):
    n_points = pca_coords.shape[0]
    n_arch = archetypes.shape[0]
    probs = np.zeros((n_points, n_arch))
    A_mat = np.vstack([archetypes.T, np.ones(n_arch)])

    for i in range(n_points):
        b = np.array([pca_coords[i, 0], pca_coords[i, 1], 1.0])
        alpha = np.linalg.lstsq(A_mat, b, rcond=None)[0]
        alpha = np.maximum(alpha, 0) #remove negative values
        s = np.sum(alpha)
        if s > 0:
            alpha = alpha / s
        else:
            alpha = np.ones(n_arch) / n_arch
        probs[i] = alpha

    return probs


def predict_archetype_loocv(df, metadata_cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    feature_cols = [c for c in df.columns if c not in metadata_cols and c != 'Dominant_archetype']
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df['Dominant_archetype'].values)
    n = len(df)

    models = {
        'LogReg': LogisticRegression(max_iter=2000, random_state=0, class_weight='balanced'),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(50,), random_state=0),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0, eval_metric='mlogloss',
                                 objective='multi:softmax', num_class=len(le.classes_)),
    }

    results = {}
    for name, model in models.items():
        preds = np.empty(n, dtype=int)
        scaler = StandardScaler()
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            X_test = X[i:i+1]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_clone = model.__class__(**model.get_params())
            if name == 'MLP':
                classes = np.unique(y_train)
                cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
                sample_weights = np.array([cw[v] for v in y_train])
                model_clone.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            elif name == 'XGBoost':
                classes = np.unique(y_train)
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                sample_weights = np.array([cw[list(classes).index(v)] for v in y_train])
                model_clone.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model_clone.fit(X_train_scaled, y_train)
            preds[i] = model_clone.predict(X_test_scaled)[0]
        preds_orig = le.inverse_transform(preds)
        y_orig = le.inverse_transform(y)
        acc = np.mean(preds_orig == y_orig)
        results[name] = {'predictions': preds_orig, 'accuracy': acc}

    return results
