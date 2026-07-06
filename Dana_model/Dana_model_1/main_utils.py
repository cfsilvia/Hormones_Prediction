import pandas as pd
import numpy as np
import os

# input: directory_path (str) — path to folder with Excel files
 #output: behavior_every_day_df (DataFrame), hormones_df (DataFrame)
def load_data(directory_path):
    behavior_every_day_file = 'Behavior_every_day.xlsx'
    hormones = 'Hormones.xlsx'
    behavior_every_day_df = pd.read_excel(f'{directory_path}/{behavior_every_day_file}')
    hormones_df = pd.read_excel(f'{directory_path}/{hormones}')
    return behavior_every_day_df, hormones_df

# input: behavior_cols (list of str), directory_path (str)
# output: None (saves behavior_parameters.xlsx)
def save_behavior_parameters(behavior_cols, directory_path):
    behavior_params_df = pd.DataFrame({'behavior_parameter': behavior_cols})
    behavior_params_path = os.path.join(directory_path, 'behavior_parameters.xlsx')
    behavior_params_df.to_excel(behavior_params_path, index=False)
    print(f'  Saved behavior parameters: {behavior_params_path}')

# input: None
# output: state (dict) — initial loop state with counters, accumulators, seen_indices
def setup_iteration_state():
    return dict(
        iteration=0, attempts=0, tables=[], best_f1=0.48, best_iter=0,
        if_dominant_archetype=False, reference_archetypes=None,
        all_true=[], all_preds_by_model=[], seen_indices=set(),
        iteration_metrics=[], aggregated_shap={}, shap_iter_count=0, feature_cols=None,
    )

# input: behavior_df (DataFrame), metadata_cols (list), all_pca_coords (ndarray), prob_coeffs (ndarray)
# output: table_df (DataFrame) — metadata + PC1/PC2 + archetype probabilities + dominant(gives 1,2,3) archetype label
def build_table_df(behavior_df, metadata_cols, all_pca_coords, prob_coeffs):
    from archetype_analysis import compute_archetype_probabilities
    table_df = behavior_df[metadata_cols].copy()
    table_df['PC1'] = all_pca_coords[:, 0]
    table_df['PC2'] = all_pca_coords[:, 1]
    table_df['Archetype1_prob'] = prob_coeffs[:, 0]
    table_df['Archetype2_prob'] = prob_coeffs[:, 1]
    table_df['Archetype3_prob'] = prob_coeffs[:, 2]
    prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
    table_df['Dominant_archetype'] = table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    return table_df

# input: table_df (DataFrame), archetypes (ndarray), hormones_df (DataFrame), if_dominant_archetype (bool)
# output: hormones_arch (DataFrame) — balanced per-animal archetypes merged with hormones
def select_top_and_merge_hormones(table_df, archetypes, hormones_df, if_dominant_archetype):
    from archetype_analysis import select_top_per_archetype, select_top_per_archetype_dominant
    if if_dominant_archetype:
        top_df = select_top_per_archetype_dominant(table_df)
    else:
        top_df = select_top_per_archetype(table_df, archetypes)
    hormones_arch = top_df.merge(
        hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
        on=['Experiment', 'sex', 'Hierarchy'], how='inner'
    )
    drop_cols = ['n_days', 'cum_arch1', 'cum_arch2', 'cum_arch3']
    hormones_arch = hormones_arch.drop(columns=[c for c in drop_cols if c in hormones_arch.columns])
    return hormones_arch

# input: hormones_arch (DataFrame), metadata_cols (list), if_dominant_archetype (bool), mapping (ndarray), model_names (list/None)
# output: loocv_results (dict), aligned_true (list) — model predictions + aligned labels
def predict_and_align(hormones_arch, metadata_cols, if_dominant_archetype, mapping, model_names=None):
    from archetype_analysis import predict_archetype_loocv
    from alignment.label_alignment import apply_mapping
    loocv_results = predict_archetype_loocv(hormones_arch, metadata_cols, model_names=model_names)
    if not if_dominant_archetype:
        for mname, mres in loocv_results.items():
            mres['predictions'] = apply_mapping(mres['predictions'], mapping)
        aligned_true = apply_mapping(hormones_arch['Dominant_archetype'].values, mapping)
    else:
        aligned_true = list(hormones_arch['Dominant_archetype'].values)
    return loocv_results, aligned_true

# input: aligned_true (list), loocv_results (dict)
# output: rows (list of dict) — accuracy, F1, precision, recall per model
def compute_iteration_metrics(aligned_true, loocv_results):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rows = []
    for mname, mres in loocv_results.items():
        acc = accuracy_score(aligned_true, mres['predictions'])
        f1_macro = f1_score(aligned_true, mres['predictions'], average='macro')
        prec_per = precision_score(aligned_true, mres['predictions'], average=None)
        rec_per = recall_score(aligned_true, mres['predictions'], average=None)
        f1_per = f1_score(aligned_true, mres['predictions'], average=None)
        rows.append(dict(model=mname, accuracy=acc, f1_macro=f1_macro,
            precision_c1=prec_per[0], precision_c2=prec_per[1], precision_c3=prec_per[2],
            recall_c1=rec_per[0], recall_c2=rec_per[1], recall_c3=rec_per[2],
            f1_c1=f1_per[0], f1_c2=f1_per[1], f1_c3=f1_per[2]))
    return rows

# input: hormones_arch (DataFrame), metadata_cols (list)
# output: list of str — hormone feature column names
def get_feature_cols(hormones_arch, metadata_cols):
    return [c for c in hormones_arch.columns
            if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2']

# input: shap_results (dict), feature_cols (list), aggregated_shap (dict), pvalues (dict)
# output: aggregated_shap (dict) — updated with per-iteration SHAP values
def process_shap_results(shap_results, feature_cols, aggregated_shap, pvalues):
    from sklearn.preprocessing import StandardScaler
    X_full = feature_cols[0] if isinstance(feature_cols, np.ndarray) else None
    for m in shap_results:
        shap_vals = shap_results[m]
        if isinstance(shap_vals, list):
            shap_vals = np.array(shap_vals).transpose(1, 2, 0)
        per_feature_mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
        per_class_mean_abs = np.mean(np.abs(shap_vals), axis=0)
        if m not in aggregated_shap:
            aggregated_shap[m] = {'mean_abs': [], 'per_class_mean_abs': [],
                                  'p_values': [], 'raw_shap': [], 'raw_features': []}
        aggregated_shap[m]['mean_abs'].append(per_feature_mean_abs)
        aggregated_shap[m]['per_class_mean_abs'].append(per_class_mean_abs)
        aggregated_shap[m]['p_values'].append(pvalues[m])
        aggregated_shap[m]['raw_shap'].append(shap_vals)
        aggregated_shap[m]['raw_features'].append(X_full if m == 'XGBoost' else None)
    return aggregated_shap

# input: iteration_metrics (list of dict), all_true (list), all_preds_by_model (list), directory_path (str)
# output: (df_iter, df_agg) — saves per_iteration + aggregated sheets to Excel
def save_classification_metrics(iteration_metrics, all_true, all_preds_by_model, directory_path):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    df_iter = pd.DataFrame(iteration_metrics)
    agg_rows = []
    for mname in df_iter['model'].unique():
        yt = np.concatenate(all_true)
        yp = np.concatenate([entry[mname] for entry in all_preds_by_model])
        acc = accuracy_score(yt, yp)
        f1_macro = f1_score(yt, yp, average='macro')
        prec_per = precision_score(yt, yp, average=None)
        rec_per = recall_score(yt, yp, average=None)
        f1_per = f1_score(yt, yp, average=None)
        agg_rows.append(dict(model=mname, accuracy=acc, f1_macro=f1_macro,
            precision_c1=prec_per[0], precision_c2=prec_per[1], precision_c3=prec_per[2],
            recall_c1=rec_per[0], recall_c2=rec_per[1], recall_c3=rec_per[2],
            f1_c1=f1_per[0], f1_c2=f1_per[1], f1_c3=f1_per[2]))
    df_agg = pd.DataFrame(agg_rows)
    excel_path = os.path.join(directory_path, 'classification_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_iter.to_excel(writer, sheet_name='per_iteration', index=False)
        df_agg.to_excel(writer, sheet_name='aggregated', index=False)
    print(f'  Saved classification metrics: {excel_path}')
    return df_iter, df_agg

# input: aggregated_shap (dict), feature_cols (list), shap_iter_count (int), directory_path (str)
# output: None (saves shap_aggregated.xlsx)
def save_aggregated_shap(aggregated_shap, feature_cols, shap_iter_count, directory_path):
    if not aggregated_shap:
        return
    shap_agg_rows = []
    for mname, mdata in aggregated_shap.items():
        mean_p = np.mean(mdata['p_values'])
        if mean_p >= 0.05:
            continue
        mean_abs_arr = np.array(mdata['mean_abs'])
        avg_mean_abs = np.mean(mean_abs_arr, axis=0)
        std_mean_abs = np.std(mean_abs_arr, axis=0)
        for i, feat in enumerate(feature_cols):
            shap_agg_rows.append(dict(
                model=mname, feature=feat,
                mean_abs_shap=avg_mean_abs[i],
                std_abs_shap=std_mean_abs[i],
                mean_p_value=mean_p,
                n_iterations=shap_iter_count
            ))
    df_shap = pd.DataFrame(shap_agg_rows)
    shap_excel_path = os.path.join(directory_path, 'shap_aggregated.xlsx')
    with pd.ExcelWriter(shap_excel_path, engine='openpyxl') as writer:
        df_shap.to_excel(writer, sheet_name='shap_aggregated', index=False)
    print(f'  Saved aggregated SHAP: {shap_excel_path}')

# input: all_true (list of lists), all_preds_by_model (list of dicts)
# output: agg_confusion (dict) — summed confusion matrix per model
def compute_aggregate_metrics(all_true, all_preds_by_model):
    from alignment.label_alignment import compute_aggregate_confusion
    agg_confusion = compute_aggregate_confusion(all_true, all_preds_by_model)
    return agg_confusion

# input: all_true (list of lists), all_preds_by_model (list of dicts), n_classes (int)
# output: agg_cm_stats (dict) — mean, std, and SE confusion matrices per model
def compute_mean_confusion_with_se(all_true, all_preds_by_model, n_classes=3):
    from sklearn.metrics import confusion_matrix
    agg_cm_stats = {}
    
    for model_name in all_preds_by_model[0].keys():
        cm_list = []
        for i in range(len(all_true)):
            cm = confusion_matrix(all_true[i], all_preds_by_model[i][model_name],
                                 labels=list(range(1, n_classes + 1)))
            cm_list.append(cm)
        
        cm_array = np.array(cm_list)  # Shape: (n_iterations, n_classes, n_classes)
        n_iterations = len(cm_list)
        
        agg_cm_stats[model_name] = {
            'mean_cm': np.mean(cm_array, axis=0),
            'std_cm': np.std(cm_array, axis=0),
            'se_cm': np.std(cm_array, axis=0) / np.sqrt(n_iterations),
            'n_iterations': n_iterations
        }
    
    return agg_cm_stats


def compute_mean_normalized_confusion_with_se(all_true, all_preds_by_model, n_classes=3, normalize='true'):
    """
    Compute mean and standard error of per-iteration normalized confusion matrices.

    Parameters:
    - all_true: list of true-label arrays per iteration
    - all_preds_by_model: list of dicts of predictions per iteration
    - n_classes: number of classes
    - normalize: one of {'true', 'pred', 'all', None} indicating normalization per iteration:
        'true' -> row-wise (true label), 'pred' -> column-wise, 'all' -> global,
        None -> no normalization (raw counts)

    Returns:
    - dict keyed by model name with 'mean_cm', 'std_cm', 'se_cm', 'n_iterations'
      where mean_cm contains the mean of the normalized matrices across iterations.
    """
    from sklearn.metrics import confusion_matrix
    agg_norm_stats = {}

    for model_name in all_preds_by_model[0].keys():
        cm_norm_list = []
        for i in range(len(all_true)):
            cm = confusion_matrix(all_true[i], all_preds_by_model[i][model_name],
                                  labels=list(range(1, n_classes + 1)))
            cm = cm.astype(float)
            if normalize is None:
                cm_norm = cm
            elif normalize == 'true':
                row_sums = cm.sum(axis=1, keepdims=True)
                # avoid division by zero
                cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
            elif normalize == 'pred':
                col_sums = cm.sum(axis=0, keepdims=True)
                cm_norm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)
            elif normalize == 'all':
                total = cm.sum()
                cm_norm = cm / total if total > 0 else np.zeros_like(cm)
            else:
                raise ValueError("normalize must be one of {None, 'true', 'pred', 'all'}")

            cm_norm_list.append(cm_norm)

        cm_norm_array = np.array(cm_norm_list)  # (n_iter, n_classes, n_classes)
        n_iterations = cm_norm_array.shape[0]
        mean_cm = np.mean(cm_norm_array, axis=0)
        std_cm = np.std(cm_norm_array, axis=0)
        se_cm = std_cm / np.sqrt(n_iterations) if n_iterations > 0 else np.zeros_like(std_cm)

        agg_norm_stats[model_name] = {
            'mean_cm': mean_cm,
            'std_cm': std_cm,
            'se_cm': se_cm,
            'n_iterations': n_iterations,
        }

    return agg_norm_stats
