from config import ArchetypeConfig
from PersonalityPipeline import PersonalityPipeline
from PersonalityPipelineMulticlass import PersonalityPipelineMulticlass

from preprocessing.loader import BehaviorDataLoader
from preprocessing.preprocessor import BehaviorPreprocessor

from models.pca_model import PersonalityPCA
from models.archetype_model import ArchetypeModel
#from models.hormone_prediction import HormonePredictionModel
from sklearn.preprocessing import StandardScaler
from analysis.correlation_analysis import ArchetypeCorrelationAnalyzer

from visualization.plotting import PersonalityVisualizer
from PersonalityPipelineMulticlass import PersonalityPipelineMulticlass





def main():
    config = ArchetypeConfig()
    pipeline = PersonalityPipeline(config)
    filename_Data = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Data_behaviour.xlsx'
    output_file = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Archetype_Results.xlsx'
    output_dir = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT'
    hormones_data_file = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\data_to_use_complete_without_final.xlsx'
    model_behavior = True
    model_hormones = False
    
    auxiliary_file = r'U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Personality_Prediction_March_2026\Help_ChatGPT\Archetype_Results_NZ.xlsx'
    
    if model_behavior:
        # Run behavior analysis
        if auxiliary_file is not None:
           index_parameters = pipeline.load_index_parameters(auxiliary_file)
        behavior_results = pipeline.run_behavior_analysis(filename_Data, output_dir, index_parameters=index_parameters)
        pipeline.save_results_to_excel(output_file, behavior_results)
    elif model_hormones:
            # Run hormone prediction analysis
            hormone_behaviour_data = pipeline.join_hormone_behavior(hormones_data_file, output_dir)
            X_hormones = hormone_behaviour_data.iloc[:, 8:-4]
            meta_data_hormones = hormone_behaviour_data.iloc[:, :8]
            #do pca on hormones and do clustering
           # X_hormones_scaled = StandardScaler().fit_transform(X_hormones)
            # pca = PersonalityPCA(n_components=2)
            # #optimize PCA and clusters
            # pca_results = pca.optimize_pca_and_clusters(X_hormones,  pca_range=range(2, 16),cluster_range=range(2, 7),random_state=42)
            # pca_results.to_excel(f"{output_dir}/PCA_Optimization_Results_hormones.xlsx", index=False)
            # print(f"\nSaved PCA results to: {output_dir}/PCA_Optimization_Results_hormones.xlsx")
            # pca.plot_pca_clusters(X_hormones, 2, 3, random_state=42,meta_data=meta_data_hormones)
            #################
            Y_prob = hormone_behaviour_data.iloc[:, -4]
            pipeline_m = PersonalityPipelineMulticlass()
            hormone_results = pipeline_m.run_prediction_multiclass(X_hormones, Y_prob, output_dir, meta_data=meta_data_hormones)
           
           



if __name__ == "__main__":

    main()