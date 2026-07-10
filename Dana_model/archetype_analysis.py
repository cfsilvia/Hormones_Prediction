import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from py_pcha import PCHA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import LeaveOneOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import zscore
from main_utils import build_table_df, select_top_and_merge_hormones


# input: df (DataFrame)
# output: df (DataFrame) — column names with spaces replaced by dots
def _normalize_column_names(df):
    df = df.copy()
    df.columns = [c.replace(' ', '.') for c in df.columns]
    return df

# input: pca_coords (ndarray), archetypes (ndarray)
# output: counts (list of int) — number of points assigned to each archetype vertex
def assign_to_nearest_vertex(pca_coords, archetypes):
    dists = np.linalg.norm(pca_coords[:, np.newaxis, :] - archetypes[np.newaxis, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    counts = [int(np.sum(assignments == v)) for v in range(archetypes.shape[0])]
    return counts

# input: data_df (DataFrame) — includes metadata and behavior columns
# output: (pca_model, pca_coords, behavior_cols, pca_metadata_df) — fitted PCA, 2D coordinates, behavior columns, metadata + PCs
def compute_pca(data_df):
    data_df = _normalize_column_names(data_df)
    exclude_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal', 'Days']
    behavior_cols = [c for c in data_df.columns if c not in exclude_cols]
    behavior_data = data_df[behavior_cols].select_dtypes(include=[np.number]).values
    behavior_data = zscore(behavior_data, nan_policy='omit')
    behavior_data = np.nan_to_num(behavior_data)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(behavior_data)
    metadata_cols = [c for c in exclude_cols if c in data_df.columns]
    pca_metadata_df = data_df[metadata_cols].copy()
    pca_metadata_df['PC1'] = pca_coords[:, 0]
    pca_metadata_df['PC2'] = pca_coords[:, 1]
    return pca, pca_coords, behavior_cols, pca_metadata_df

# input: pca (PCA model), data_df (DataFrame), behavior_cols (list of str)
# output: pca_coords (ndarray) — transformed 2D coordinates for new data
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

# input: pca_coords_all (ndarray), sample_frac (float)
# output: (pca_coords, archetypes, varexlp, counts, sample_indices)
#         — sampled coords, 3 archetype vertices, variance explained, counts, indices used
def sample_and_fit_archetypes(pca_coords_all, sample_frac=0.8, no_pca_data = None):

    if no_pca_data is not None:
        coords_to_remove = no_pca_data[['PC1','PC2']].to_numpy(dtype=float)
        matches = np.isclose(pca_coords_all[:, None, :],coords_to_remove[None, :, :]).all(axis=2).any(axis=1)
        pca_coords_filtered = pca_coords_all[~matches]
        pca_coord_all = pca_coords_filtered


    n = pca_coords_all.shape[0]
    n_sample = max(int(n * sample_frac), 3)
    sample_indices = np.random.choice(n, n_sample, replace=False)
    pca_coords = pca_coords_all[sample_indices]

    X_for_pcha = pca_coords.T
    XC, S, C, SSE, varexlp = PCHA(X_for_pcha, 3)
    archetypes = np.array(XC.T)

    counts = assign_to_nearest_vertex(pca_coords_all, archetypes)

    return pca_coords, archetypes, varexlp, counts, sample_indices

# input: pca_coords (ndarray), archetypes (ndarray)
# output: probs (ndarray) — [n_points x 3] convex combination weights for each archetype
def compute_archetype_probabilities(pca_coords, archetypes):
    n_points = pca_coords.shape[0]
    n_arch = archetypes.shape[0]
    probs = np.zeros((n_points, n_arch))
    A_mat = np.vstack([archetypes.T, np.ones(n_arch)])

    for i in range(n_points):
        b = np.array([pca_coords[i, 0], pca_coords[i, 1], 1.0])
        alpha = np.linalg.lstsq(A_mat, b, rcond=None)[0]
        alpha = np.maximum(alpha, 0)
        s = np.sum(alpha)
        if s > 0:
            alpha = alpha / s
        else:
            alpha = np.ones(n_arch) / n_arch
        probs[i] = alpha

    return probs

# input: n_classes (int), model_names (list/None)
# output: models (dict) - classifier name to initialized model
def _get_models(n_classes, model_names=None):
    models = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', max_iter=10000),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(30,), alpha=0.1),
        'RandomForest': RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=0),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            #  tree_method='hist',
            # device='cuda',
            random_state=0,
            eval_metric='mlogloss',
            objective='multi:softmax',
            num_class=n_classes
        ), 
    #    'HistGB': XGBClassifier(
    #         n_estimators=100,
    #         max_depth=2,
    #         learning_rate=0.05,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         reg_alpha=1.0,
    #         reg_lambda=1.0,
    #         min_child_weight=5,
    #         random_state=0,
    #         eval_metric='mlogloss',
    #         objective='multi:softprob',
    #         num_class=n_classes
    #     ),

        'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
        # 'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
        #                          eval_metric='mlogloss', objective='multi:softmax', num_class=n_classes),
    }
    if model_names is not None:
        unknown = [name for name in model_names if name not in models]
        if unknown:
            raise ValueError(f'Unknown model names: {unknown}')
        models = {name: models[name] for name in model_names}
    return models


# input: df (DataFrame), metadata_cols (list of str), model_names (list/None)
# output: results (dict) — per model: predictions list + accuracy from LOOCV
def predict_archetype_loocv(df, metadata_cols, model_names=None):
    feature_cols = [c for c in df.columns if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2'
                    and c not in ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']]
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df['Dominant_archetype'].values)
    n = len(df)
    n_classes = len(le.classes_)

    models = _get_models(n_classes, model_names=model_names)

    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        preds = np.empty(n, dtype=int)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            model_clone = model.__class__(**model.get_params())

            if name in ('HistGB', 'XGBoost'):
                classes = np.unique(y_train)
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                sample_weights = np.array([cw[list(classes).index(v)] for v in y_train])
                model_clone.fit(X_train, y_train, sample_weight=sample_weights)
                preds[test_idx[0]] = model_clone.predict(X_test)[0]
            elif name in ('RandomForest', 'ExtraTrees'):
                model_clone.fit(X_train, y_train)
                preds[test_idx[0]] = model_clone.predict(X_test)[0]
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                if name == 'MLP':
                    classes = np.unique(y_train)
                    cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
                    sample_weights = np.array([cw[v] for v in y_train])
                    model_clone.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                else:
                    model_clone.fit(X_train_scaled, y_train)
                preds[test_idx[0]] = model_clone.predict(X_test_scaled)[0]

        preds_orig = le.inverse_transform(preds)
        y_orig = le.inverse_transform(y)
        acc = np.mean(preds_orig == y_orig)
        results[name] = {'predictions': preds_orig, 'accuracy': acc}

    return results

# input: table_df (DataFrame), archetypes (ndarray or None), n_per_arch (int)
# output: agg (DataFrame) — one row per animal with dominant archetype, balanced via Hungarian
def select_top_per_archetype(table_df, archetypes=None, n_per_arch=20):
    group_keys = ['Experiment', 'sex', 'Hierarchy']

    grouped = table_df.groupby(group_keys)
    agg = grouped.agg(
        PC1=('PC1', 'mean'),
        PC2=('PC2', 'mean'),
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    ).reset_index()

    cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
    agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    if archetypes is not None:
        mean_pc = agg[['PC1', 'PC2']].values
        n_animals = len(agg)
        n_arch = archetypes.shape[0]
        total_slots = n_per_arch * n_arch
        prob_cols = [f'Archetype{i + 1}_prob' for i in range(n_arch)]

        arch_probs = compute_archetype_probabilities(mean_pc, archetypes)
        agg[prob_cols] = arch_probs
 
        # This makes the assignment prefer animals with higher mean-PCA probability for each archetype.
        cost_matrix = np.zeros((n_animals, total_slots))
        for j in range(n_arch):
            for k in range(n_per_arch):
                cost_matrix[:, j * n_per_arch + k] = -arch_probs[:, j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        selected_archetypes = col_ind // n_per_arch

        agg = agg.iloc[row_ind].copy()
        agg['Dominant_archetype'] = selected_archetypes + 1

    return agg

# input: table_df (DataFrame)
# output: agg (DataFrame) — one row per animal, dominant archetype by max cum probability
def select_top_per_archetype_dominant(table_df):
    group_keys = ['Experiment', 'sex', 'Hierarchy']

    grouped = table_df.groupby(group_keys)
    agg = grouped.agg(
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    ).reset_index()

    cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
    agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    return agg

# input: table_df (DataFrame), archetypes (ndarray or None), n_per_arch (int)
# output: agg (DataFrame) — one row per animal with dominant archetype, balanced via Hungarian
def select_top_per_archetype_with_distance(table_df, archetypes=None, n_per_arch=20):
    group_keys = ['Experiment', 'sex', 'Hierarchy']

    grouped = table_df.groupby(group_keys)
    agg = grouped.agg(
        PC1=('PC1', 'mean'),
        PC2=('PC2', 'mean'),
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    ).reset_index()

    cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
    agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    if archetypes is not None:
        mean_pc = agg[['PC1', 'PC2']].values
        n_animals = len(agg)
        n_arch = archetypes.shape[0]
        total_slots = n_per_arch * n_arch

        dists = np.zeros((n_animals, n_arch))
        for j in range(n_arch):
            dists[:, j] = np.linalg.norm(mean_pc - archetypes[j], axis=1)

        cost_matrix = np.zeros((n_animals, total_slots))
        for j in range(n_arch):
            for k in range(n_per_arch):
                cost_matrix[:, j * n_per_arch + k] = dists[:, j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        selected_archetypes = col_ind // n_per_arch

        agg = agg.iloc[row_ind].copy()
        agg['Dominant_archetype'] = selected_archetypes + 1

    return agg

# input: mean_coords_df, behavior_df, metadata_cols, all_pca_coords, hormones_df, if_dominant_archetype, directory_path
# output: (mean_table_df, mean_hormones_arch) - per-day probabilities and animal-level archetype/hormone table
def build_mean_archetype_assignment(mean_coords_df, behavior_df, metadata_cols, all_pca_coords, hormones_df, if_dominant_archetype=False, directory_path=None):
    # ------------------------------------------------------------------
    # Mean archetype coordinates
    # ------------------------------------------------------------------
    mean_archetypes = (mean_coords_df[['PC1', 'PC2']].to_numpy(dtype=float))

    # ------------------------------------------------------------------
    # Probability of every day belonging to the mean archetypes
    # ------------------------------------------------------------------
    mean_prob_coeffs = compute_archetype_probabilities(all_pca_coords, mean_archetypes)
 
    # ------------------------------------------------------------------
    # Build the standard per-day table
    # ------------------------------------------------------------------
    mean_table_df = build_table_df(behavior_df, metadata_cols, all_pca_coords, mean_prob_coeffs)

    # ------------------------------------------------------------------
    # Aggregate animals and merge hormones
    # ------------------------------------------------------------------
    mean_hormones_arch = select_top_and_merge_hormones(mean_table_df, mean_archetypes, hormones_df, if_dominant_archetype)

    if directory_path is not None:
        mean_table_df.to_excel(os.path.join(directory_path, 'mean_archetype_probabilities_per_day.xlsx'), index=False)
        mean_hormones_arch.to_excel(os.path.join(directory_path, 'mean_hormones_with_archetypes.xlsx'), index=False)

    return mean_table_df, mean_hormones_arch


# input: label (scalar-like)
# output: label key (str) safe for metrics dictionaries
def _as_label_key(label):
    return str(label.item() if hasattr(label, 'item') else label)


# input: values (array-like)
# output: sigmoid-transformed values (ndarray)
def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


# input: values (array-like)
# output: normalized softmax probabilities (ndarray)
def _softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = np.sum(exp_values)
    if total == 0:
        return np.ones_like(values) / len(values)
    return exp_values / total


# input: model (fitted estimator), X_test (ndarray), n_classes (int), pred_class (int)
# output: scores (ndarray) - one probability/score per class for one test sample
def _predict_class_scores(model, X_test, n_classes, pred_class):
    if hasattr(model, 'predict_proba'):
        try:
            raw_scores = np.asarray(model.predict_proba(X_test))[0]
            scores = np.zeros(n_classes, dtype=float)
            classes = getattr(model, 'classes_', np.arange(len(raw_scores)))
            for score_idx, class_idx in enumerate(classes):
                if int(class_idx) < n_classes:
                    scores[int(class_idx)] = raw_scores[score_idx]
            return scores
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        try:
            raw_scores = np.asarray(model.decision_function(X_test))
            raw_scores = raw_scores[0] if raw_scores.ndim > 1 else raw_scores
            classes = getattr(model, 'classes_', np.arange(n_classes))

            if raw_scores.ndim == 0 or raw_scores.shape[0] == 1:
                prob_positive = float(np.ravel(_sigmoid(raw_scores))[0])
                scores = np.zeros(n_classes, dtype=float)
                if len(classes) == 2:
                    scores[int(classes[0])] = 1.0 - prob_positive
                    scores[int(classes[1])] = prob_positive
                else:
                    scores[pred_class] = 1.0
                return scores

            scores = np.full(n_classes, np.min(raw_scores) - 1.0, dtype=float)
            for score_idx, class_idx in enumerate(classes):
                if int(class_idx) < n_classes:
                    scores[int(class_idx)] = raw_scores[score_idx]
            return _softmax(scores)
        except Exception:
            pass

    scores = np.zeros(n_classes, dtype=float)
    scores[pred_class] = 1.0
    return scores


# input: y_true, y_pred, class_scores, class_labels
# output: metrics (dict) - macro F1, per-class F1/precision/recall, and one-vs-rest ROC AUC
def _compute_classification_metrics(y_true, y_pred, class_scores, class_labels):
    labels = np.arange(len(class_labels))
    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    roc_auc_per_class = []
    for class_idx in labels:
        binary_true = (y_true == class_idx).astype(int)
        if len(np.unique(binary_true)) < 2:
            roc_auc_per_class.append(np.nan)
        else:
            roc_auc_per_class.append(roc_auc_score(binary_true, class_scores[:, class_idx]))
    roc_auc_per_class = np.array(roc_auc_per_class, dtype=float)
    roc_auc_macro = np.nanmean(roc_auc_per_class) if np.any(~np.isnan(roc_auc_per_class)) else np.nan

    label_keys = [_as_label_key(label) for label in class_labels]
    return {
        'f1_macro': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'f1_per_class': dict(zip(label_keys, f1_per_class.tolist())),
        'precision_per_class': dict(zip(label_keys, precision_per_class.tolist())),
        'recall_per_class': dict(zip(label_keys, recall_per_class.tolist())),
        'roc_auc_ovr_macro': roc_auc_macro,
        'roc_auc_per_class': dict(zip(label_keys, roc_auc_per_class.tolist())),
    }



# input: shap_values, n_samples (int), n_features (int), n_classes (int)
# output: shap_array (ndarray) normalized to samples x features x classes when possible
def _normalize_shap_values(shap_values, n_samples, n_features, n_classes):
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=2)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 2:
        return shap_array
    if shap_array.ndim != 3:
        raise ValueError(f'Unsupported SHAP values shape: {shap_array.shape}')

    if shap_array.shape[0] == n_samples and shap_array.shape[1] == n_features:
        return shap_array
    if shap_array.shape[0] == n_classes and shap_array.shape[1] == n_samples and shap_array.shape[2] == n_features:
        return np.transpose(shap_array, (1, 2, 0))
    if shap_array.shape[0] == n_samples and shap_array.shape[2] == n_features:
        return np.transpose(shap_array, (0, 2, 1))
    return shap_array


# input: shap_values (ndarray), pred_class (int)
# output: pred_shap (ndarray) - feature SHAP values for the predicted class
def _predicted_class_shap(shap_values, pred_class):
    if shap_values.ndim == 2:
        return shap_values[0]
    return shap_values[0, :, pred_class]


# input: model, X_background (ndarray), X_test (ndarray), model_name (str), n_classes (int)
# output: (shap_values, expected_value) for the test sample
def _compute_shap_for_test_sample(model, X_background, X_test, model_name, n_classes):
    try:
        import shap
    except ImportError as exc:
        raise ImportError('predict_archetype_loocv_with_shap requires the shap package') from exc

    if model_name in ('RandomForest', 'ExtraTrees', 'XGBoost'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_value = getattr(explainer, 'expected_value', None)
    elif model_name in ('LogReg', 'SVM'):
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X_test)
        expected_value = getattr(explainer, 'expected_value', None)
    else:
        explainer = shap.PermutationExplainer(model.predict_proba, X_background)
        explanation = explainer(X_test, max_evals=2 * X_background.shape[1] + 1)
        shap_values = explanation.values
        expected_value = explanation.base_values

    return _normalize_shap_values(shap_values, X_test.shape[0], X_test.shape[1], n_classes), expected_value


# input: df (DataFrame), metadata_cols (list of str), model_names (list/None), top_n (int/None)
# output: results (dict) - per model LOOCV predictions, metrics, SHAP values, metadata, and explanations
def predict_archetype_loocv_with_shap(df, metadata_cols, model_names=None, top_n=None):
    metadata_cols = [c for c in metadata_cols if c != 'Days']
    feature_cols = [c for c in df.columns if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2'
                    and c not in ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']]
    feature_df = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = list(feature_df.columns)
    X = feature_df.values
    le = LabelEncoder()
    y = le.fit_transform(df['Dominant_archetype'].values)
    n = len(df)
    n_classes = len(le.classes_)
    class_labels = le.classes_

    models = _get_models(n_classes, model_names=model_names)

    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        preds = np.empty(n, dtype=int)
        class_scores = np.zeros((n, n_classes), dtype=float)
        shap_pred_class = np.empty((n, len(feature_cols)), dtype=float)
        shap_all_classes = []
        expected_values = []
        explanations = []

        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
            test_pos = test_idx[0]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            model_clone = model.__class__(**model.get_params())

            if name in ('HistGB', 'XGBoost'):
                classes = np.unique(y_train)
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                sample_weights = np.array([cw[list(classes).index(v)] for v in y_train])
                model_clone.fit(X_train, y_train, sample_weight=sample_weights)
                preds[test_pos] = model_clone.predict(X_test)[0]
                class_scores[test_pos] = _predict_class_scores(model_clone, X_test, n_classes, preds[test_pos])
                shap_background = X_train
                shap_test = X_test
            elif name in ('RandomForest', 'ExtraTrees'):
                model_clone.fit(X_train, y_train)
                preds[test_pos] = model_clone.predict(X_test)[0]
                class_scores[test_pos] = _predict_class_scores(model_clone, X_test, n_classes, preds[test_pos])
                shap_background = X_train
                shap_test = X_test
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                if name == 'MLP':
                    classes = np.unique(y_train)
                    cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
                    sample_weights = np.array([cw[v] for v in y_train])
                    model_clone.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                else:
                    model_clone.fit(X_train_scaled, y_train)
                preds[test_pos] = model_clone.predict(X_test_scaled)[0]
                class_scores[test_pos] = _predict_class_scores(model_clone, X_test_scaled, n_classes, preds[test_pos])
                shap_background = X_train_scaled
                shap_test = X_test_scaled

            shap_values, expected_value = _compute_shap_for_test_sample(
                model_clone, shap_background, shap_test, name, n_classes,
            )
            pred_shap = _predicted_class_shap(shap_values, preds[test_pos])
            shap_pred_class[test_pos] = pred_shap
            shap_all_classes.append(shap_values[0] if shap_values.ndim == 3 else shap_values[0])
            expected_values.append(expected_value)

            order = np.argsort(np.abs(pred_shap))[::-1]
            if top_n is not None:
                order = order[:top_n]
            explanations.append({
                'iteration': fold_idx,
                'row_index': df.index[test_pos],
                'true_label': le.inverse_transform([y[test_pos]])[0],
                'predicted_label': le.inverse_transform([preds[test_pos]])[0],
                'top_features': [feature_cols[i] for i in order],
                'top_shap_values': pred_shap[order].tolist(),
                'top_feature_values': X_test[0, order].tolist(),
            })

        preds_orig = le.inverse_transform(preds)
        y_orig = le.inverse_transform(y)
        acc = np.mean(preds_orig == y_orig)
        metrics = _compute_classification_metrics(y, preds, class_scores, class_labels)
        results[name] = {
            'predictions': preds_orig,
            'true_labels': y_orig,
            'metadata': df[metadata_cols].copy(),
            'accuracy': acc,
            'f1_macro': metrics['f1_macro'],
            'f1_per_class': metrics['f1_per_class'],
            'precision_per_class': metrics['precision_per_class'],
            'recall_per_class': metrics['recall_per_class'],
            'roc_auc': metrics['roc_auc_ovr_macro'],
            'roc_auc_ovr_macro': metrics['roc_auc_ovr_macro'],
            'roc_auc_per_class': metrics['roc_auc_per_class'],
            'class_scores': class_scores,
            'feature_cols': feature_cols,
            'shap_values': shap_pred_class,
            'shap_values_all_classes': np.array(shap_all_classes, dtype=object),
            'expected_values': expected_values,
            'explanations': pd.DataFrame(explanations),
        }

    return results
