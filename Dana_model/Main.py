import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from archetype_analysis import compute_pca, sample_and_fit_archetypes, compute_archetype_probabilities, assign_to_nearest_vertex, predict_archetype_loocv, select_top_per_archetype, select_top_per_archetype_dominant
from alignment.label_alignment import get_alignment_mapping, apply_mapping, accumulate_results, compute_aggregate_confusion
from plot_utils import (setup_figure, draw_triangle, plot_confusion_matrices, save_if_best,
    plot_aggregate_confusion, plot_permutation_tests, plot_aggregate_permutation,
    compute_shap_values, plot_shap_feature_importance, plot_shap_per_archetype)
from shap_calculation import train_full_models_and_shap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import os
import openpyxl


def main():
    directory_path = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\June_pareto_all_data\Dana_model\data_to_use'
    behavior_every_day_file = 'Behavior_every_day.xlsx'
    hormones = 'Hormones.xlsx'

    behavior_every_day_df = pd.read_excel(f'{directory_path}/{behavior_every_day_file}')
    hormones_df = pd.read_excel(f'{directory_path}/{hormones}')

    min_per_vertex = 40
    #do pca on the behavior data and get the PCA coordinates for all samples
    pca_model, all_pca_coords, behavior_cols = compute_pca(behavior_every_day_df)
    #save the behavior columns in an excel file for reference
    behavior_params_df = pd.DataFrame({'behavior_parameter': behavior_cols})
    behavior_params_path = os.path.join(directory_path, 'behavior_parameters.xlsx')
    behavior_params_df.to_excel(behavior_params_path, index=False)
    print(f'  Saved behavior parameters: {behavior_params_path}')
    ###################################################

    metadata_cols = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Animal']

    plt.ion()
    fig, ax_tri, cm_axes, perm_axes = setup_figure()

    iteration = 0
    attempts = 0
    tables = []
    best_f1 = 0.48
    best_iter = 0
    if_dominant_archetype = False
    reference_archetypes = None
    all_true = []
    all_preds_by_model = []
    seen_indices = set()
    iteration_metrics = []
    aggregated_shap = {}
    shap_iter_count = 0
    feature_cols = None

    while state['iteration'] < 50:
        state['attempts'] += 1
        pca_coords, archetypes, varexlp, counts, sample_indices = \
            sample_and_fit_archetypes(all_pca_coords)

        idx_tuple = tuple(sorted(sample_indices))
        if idx_tuple in seen_indices:
            continue
        seen_indices.add(idx_tuple)

        # Check if the counts per vertex meet the minimum requirement 
        if all(c >= min_per_vertex for c in counts):
            iteration += 1
            if reference_archetypes is None:#for alignment, we need to set the reference archetypes in the first iteration
                reference_archetypes = archetypes.copy()
            #align the current archetypes to the reference archetypes    
            mapping = get_alignment_mapping(reference_archetypes, archetypes)
            draw_triangle(ax_tri, all_pca_coords, archetypes, counts, varexlp, iteration, attempts, mapping=mapping)

            print(f'Iteration {iteration}/50 accepted - Var explained: {varexlp:.3f}  '
                  f'Counts per vertex: {counts}')

            prob_coeffs = compute_archetype_probabilities(all_pca_coords, archetypes)
            table_df = behavior_every_day_df[metadata_cols].copy()
            table_df['PC1'] = all_pca_coords[:, 0]
            table_df['PC2'] = all_pca_coords[:, 1]
            table_df['Archetype1_prob'] = prob_coeffs[:, 0]
            table_df['Archetype2_prob'] = prob_coeffs[:, 1]
            table_df['Archetype3_prob'] = prob_coeffs[:, 2]
            prob_cols = ['Archetype1_prob', 'Archetype2_prob', 'Archetype3_prob']
            table_df['Dominant_archetype'] = table_df[prob_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
            tables.append(table_df)
            
            #select how distributed the samples of the different mice- get table for 60 mice not all the samples-mean per day of pca
            if if_dominant_archetype:
                top_df = select_top_per_archetype_dominant(table_df)
            else:
                top_df = select_top_per_archetype(table_df, archetypes)
            #join with hormones data
            hormones_arch = top_df.merge(
                hormones_df.drop_duplicates(subset=['Experiment', 'sex', 'Hierarchy']),
                on=['Experiment', 'sex', 'Hierarchy'], how='inner'
            )
            drop_cols = ['n_days', 'cum_arch1', 'cum_arch2', 'cum_arch3']
            hormones_arch = hormones_arch.drop(columns=[c for c in drop_cols if c in hormones_arch.columns])
            
            #learn models- include list of dictionary with model name and predictions for each iteration
            loocv_results = predict_archetype_loocv(hormones_arch, metadata_cols)
            
            #correct name of archetypes in the predictions according to the mapping- if not dominant archetype,
            if not if_dominant_archetype:
                for mname, mres in loocv_results.items():
                    mres['predictions'] = apply_mapping(mres['predictions'], mapping)
                aligned_true = apply_mapping(hormones_arch['Dominant_archetype'].values, mapping)
            else:
                aligned_true = list(hormones_arch['Dominant_archetype'].values)
            #accumulate results for aggregate confusion matrix and metrics- from all iterations   
            all_true, all_preds_by_model = accumulate_results(all_true, all_preds_by_model, aligned_true, loocv_results)
            
            #print metrics for this iteration- accumulate in list of dictionaries for each iteration and model to save in excel at the end
            for mname, mres in loocv_results.items():
                print(f'  {mname} LOOCV accuracy: {mres["accuracy"]:.3f}')
                acc = accuracy_score(aligned_true, mres['predictions'])
                f1_macro = f1_score(aligned_true, mres['predictions'], average='macro')
                prec_per = precision_score(aligned_true, mres['predictions'], average=None)
                rec_per = recall_score(aligned_true, mres['predictions'], average=None)
                f1_per = f1_score(aligned_true, mres['predictions'], average=None)
                iteration_metrics.append(dict(iteration=iteration, model=mname, accuracy=acc,
                    f1_macro=f1_macro, precision_c1=prec_per[0], precision_c2=prec_per[1], precision_c3=prec_per[2],
                    recall_c1=rec_per[0], recall_c2=rec_per[1], recall_c3=rec_per[2],
                    f1_c1=f1_per[0], f1_c2=f1_per[1], f1_c3=f1_per[2]))
            
            #plot confusion matrices and permutation tests for this iteration-after mapping
            f1_scores, per_class_f1 = plot_confusion_matrices(cm_axes, loocv_results, aligned_true)
            # do permutation tests and plot the results -after mapping
            pvalues =plot_permutation_tests(perm_axes, loocv_results, aligned_true)
            shap_results = train_full_models_and_shap(hormones_arch, metadata_cols, aligned_true)
            if feature_cols is None:
                feature_cols = [c for c in hormones_arch.columns
                                if c not in metadata_cols and c != 'Dominant_archetype' and c != 'PC1' and c != 'PC2']
            
            X_full = hormones_arch[feature_cols].select_dtypes(include=[np.number]).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)

            for m in shap_results:
                shap_vals = shap_results[m]
                if isinstance(shap_vals, list):
                    shap_vals = np.array(shap_vals).transpose(1, 2, 0)
                per_feature_mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
                per_class_mean_abs = np.mean(np.abs(shap_vals), axis=0)
                X_raw = X_full if m == 'XGBoost' else X_scaled
                if m not in aggregated_shap:
                    aggregated_shap[m] = {'mean_abs': [], 'per_class_mean_abs': [],
                                          'p_values': [], 'raw_shap': [], 'raw_features': []}
                aggregated_shap[m]['mean_abs'].append(per_feature_mean_abs)
                aggregated_shap[m]['per_class_mean_abs'].append(per_class_mean_abs)
                aggregated_shap[m]['p_values'].append(pvalues[m])
                aggregated_shap[m]['raw_shap'].append(shap_vals)
                aggregated_shap[m]['raw_features'].append(X_raw)
            shap_iter_count += 1

            saved = save_if_best(fig, directory_path, f1_scores, per_class_f1, iteration)
           
            plt.pause(0.8)
        else:
            print(f'  Rejected (attempt {attempts}) — counts per vertex: {counts}')

    agg_confusion = compute_aggregate_confusion(all_true, all_preds_by_model)
    plot_aggregate_confusion(agg_confusion, directory_path)
    plot_aggregate_permutation(all_true, all_preds_by_model, directory_path)

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

    if aggregated_shap:
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

    if aggregated_shap:
        plot_shap_per_archetype(aggregated_shap, feature_cols, directory_path)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
