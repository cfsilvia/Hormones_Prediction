from auxiliary_files import load_data, run_archetype_analysis, correlate_with_behaviour, aggregate_and_merge_hormones, correlate_with_hormones, load_hormones_archetypes
from Auxiliary_prediction import create_archetype_prediction_data, predict_archetype_labels_loocv, calculate_prediction_metrics, plot_confusion_matrices_by_archetype
import os



def main():
    data_dir = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Dana_Noga_model_July\data_to_use'
    create_data = False
    run_model = True
    models = ["SVC_linear","random_forest","logistic", "decision_tree","k_neighbors","xgboost"]

    if create_data:
        behavior_df, hormones_df = load_data(data_dir)
        
        archetypes, all_pca_coords, prob_coeffs, table_df, metadata_cols = \
            run_archetype_analysis(behavior_df, data_dir)
        
        corr_behav = correlate_with_behaviour(behavior_df, prob_coeffs, metadata_cols, data_dir)

        hormones_arch = aggregate_and_merge_hormones(table_df, hormones_df, metadata_cols, archetypes, data_dir)

        #correlation of hormones with the probability
        corr_horm = correlate_with_hormones(hormones_arch, metadata_cols, data_dir)

    elif run_model:
        hormones_arch = load_hormones_archetypes(data_dir)
        prediction_df, hormone_cols, prediction_full_df = create_archetype_prediction_data(
            hormones_arch,
            save_path=os.path.join(data_dir, 'prediction_data.xlsx'),
            full_save_path=os.path.join(data_dir, 'prediction_data_with_metadata.xlsx')
        )
        prediction_results = predict_archetype_labels_loocv(prediction_df, hormone_cols, models)
        prediction_metrics = calculate_prediction_metrics(
            prediction_results,
            save_path=os.path.join(data_dir, 'prediction_metrics.xlsx')
        )
        plot_confusion_matrices_by_archetype(
            prediction_metrics,
            models,
            save_path=os.path.join(data_dir, 'prediction_confusion_matrices.pdf')
        )



    
if __name__ == '__main__':
    main()
