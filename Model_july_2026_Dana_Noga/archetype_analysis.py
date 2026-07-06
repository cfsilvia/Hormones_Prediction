import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from py_pcha import PCHA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import LeaveOneOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore


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
# output: (pca_model, pca_coords, behavior_cols) — fitted PCA, 2D coordinates, column names
def compute_pca(data_df):
    data_df = _normalize_column_names(data_df)
    exclude_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']
    behavior_cols = [c for c in data_df.columns if c not in exclude_cols]
    behavior_data = data_df[behavior_cols].select_dtypes(include=[np.number]).values
    behavior_data = zscore(behavior_data, nan_policy='omit')
    behavior_data = np.nan_to_num(behavior_data)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(behavior_data)
    return pca, pca_coords, behavior_cols

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
def sample_and_fit_archetypes(pca_coords_all, sample_frac=0.8):
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
        b = np.append(pca_coords[i], 1.0)
        alpha = np.linalg.lstsq(A_mat, b, rcond=None)[0]
        alpha = np.maximum(alpha, 0)
        s = np.sum(alpha)
        if s > 0:
            alpha = alpha / s
        else:
            alpha = np.ones(n_arch) / n_arch
        probs[i] = alpha

    return probs

# input: df (DataFrame), metadata_cols (list of str)
# output: results (dict) — per model: predictions list + accuracy from LOOCV
def predict_archetype_loocv(df, metadata_cols):
    feature_cols = [c for c in df.columns if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2']
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df['Dominant_archetype'].values)
    n = len(df)
    n_classes = len(le.classes_)

    models = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', max_iter=10000),
        'MLP': MLPClassifier(max_iter=2000, hidden_layer_sizes=(30,), alpha=0.1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=0,
                                 eval_metric='mlogloss', objective='multi:softmax', num_class=n_classes),
    }

    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        preds = np.empty(n, dtype=int)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            model_clone = model.__class__(**model.get_params())

            if name == 'XGBoost':
                classes = np.unique(y_train)
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                sample_weights = np.array([cw[list(classes).index(v)] for v in y_train])
                model_clone.fit(X_train, y_train, sample_weight=sample_weights)
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
    pc_cols = [c for c in ['PC1', 'PC2', 'PC3'] if c in table_df.columns]

    grouped = table_df.groupby(group_keys)
    agg_kwargs = {
        pc: (pc, 'mean') for pc in pc_cols
    }
    agg_kwargs.update(
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    )
    agg = grouped.agg(**agg_kwargs).reset_index()

    prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']

    if archetypes is not None:
        pc_cols = pc_cols[:archetypes.shape[1]]
        if len(pc_cols) != archetypes.shape[1]:
            raise ValueError(f'Expected {archetypes.shape[1]} PC columns, found {len(pc_cols)}')

        mean_pc = agg[pc_cols].values
        mean_probs = compute_archetype_probabilities(mean_pc, archetypes)
        agg[prob_cols] = mean_probs
        agg['Dominant_archetype'] = agg[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

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
    else:
        cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
        agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    return agg



# input: table_df (DataFrame), archetypes (ndarray or None), n_per_arch (int)
# output: agg (DataFrame) — one row per animal with dominant archetype, balanced via Hungarian
def select_top_per_archetype(table_df, archetypes=None, n_per_arch=20):
    group_keys = ['Experiment', 'sex', 'Hierarchy']
    pc_cols = [c for c in ['PC1', 'PC2', 'PC3'] if c in table_df.columns]

    grouped = table_df.groupby(group_keys)
    agg_kwargs = {
        pc: (pc, 'mean') for pc in pc_cols
    }
    agg_kwargs.update(
        n_days=('Animal', 'count'),
        cum_arch1=('Archetype1_prob', 'sum'),
        cum_arch2=('Archetype2_prob', 'sum'),
        cum_arch3=('Archetype3_prob', 'sum'),
    )
    agg = grouped.agg(**agg_kwargs).reset_index()

    prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']

    if archetypes is not None:
        pc_cols = pc_cols[:archetypes.shape[1]]
        if len(pc_cols) != archetypes.shape[1]:
            raise ValueError(f'Expected {archetypes.shape[1]} PC columns, found {len(pc_cols)}')

        mean_pc = agg[pc_cols].values
        mean_probs = compute_archetype_probabilities(mean_pc, archetypes)
        agg[prob_cols] = mean_probs
        agg['Dominant_archetype'] = agg[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

        n_animals = len(agg)
        n_arch = archetypes.shape[0]
        total_slots = n_per_arch * n_arch

        cost_matrix = np.zeros((n_animals, total_slots))
        for j in range(n_arch):
            for k in range(n_per_arch):
                cost_matrix[:, j * n_per_arch + k] = -mean_probs[:, j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        selected_archetypes = col_ind // n_per_arch

        agg = agg.iloc[row_ind].copy()
        agg['Dominant_archetype'] = selected_archetypes + 1
    else:
        cum_cols = ['cum_arch1', 'cum_arch2', 'cum_arch3']
        agg['Dominant_archetype'] = agg[cum_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    return agg


# input: mean_coords_df, behavior_df, metadata_cols, all_pca_coords, hormones_df
# output: (mean_table_df, mean_hormones_arch) using mean archetypes and Hungarian assignment
def build_mean_archetype_assignment(mean_coords_df, behavior_df, metadata_cols, all_pca_coords, hormones_df, if_dominant_archetype=False):
    if 'archetype' in mean_coords_df.columns:
        mean_coords_df = mean_coords_df.sort_values('archetype')

    mean_archetypes = mean_coords_df[['PC1', 'PC2']].to_numpy(dtype=float)
    mean_prob_coeffs = compute_archetype_probabilities(all_pca_coords, mean_archetypes)

    mean_table_df = behavior_df[metadata_cols].copy()
    mean_table_df['PC1'] = all_pca_coords[:, 0]
    mean_table_df['PC2'] = all_pca_coords[:, 1]
    for arch_idx in range(mean_prob_coeffs.shape[1]):
        mean_table_df[f'Archetype{arch_idx + 1}_prob'] = mean_prob_coeffs[:, arch_idx]
    prob_cols = [f'Archetype{i + 1}_prob' for i in range(mean_prob_coeffs.shape[1])]
    mean_table_df['Dominant_archetype'] = mean_table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)

    mean_top_df = select_top_per_archetype(mean_table_df, mean_archetypes)

    mean_hormones_arch = mean_top_df.merge(
        hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
        on=['Experiment', 'sex', 'Hierarchy'], how='inner'
    )
    drop_cols = ['n_days', 'cum_arch1', 'cum_arch2', 'cum_arch3']
    mean_hormones_arch = mean_hormones_arch.drop(columns=[c for c in drop_cols if c in mean_hormones_arch.columns])

    return mean_table_df, mean_hormones_arch
