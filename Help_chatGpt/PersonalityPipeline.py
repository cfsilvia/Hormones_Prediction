
from dataclasses import dataclass
from tkinter.font import names
from openpyxl import load_workbook

from sklearn.preprocessing import StandardScaler

from model_evaluation.ModelEvaluator import ModelEvaluator
from model_evaluation.Permutation_test import PermutationTester
from models.Hormone_model import HormonePredictionModel
from models.archetype_model import ArchetypeModel
from preprocessing.loader import BehaviorDataLoader
from preprocessing.preprocessor import BehaviorPreprocessor
from models.pca_model import PersonalityPCA
from Results.ResultsBuilder import ResultsBuilder
from analysis.correlation_analysis import ArchetypeCorrelationAnalyzer
from visualization.plotting import PersonalityVisualizer
import pandas as pd
import matplotlib.pyplot as plt


class PersonalityPipeline:
    def __init__(self, config):
        self.config = config

    def run_behavior_analysis(self, filepath, output_dir, index_parameters=None):
            # LOAD
            loader = BehaviorDataLoader(filepath)
            df_1 = loader.load_data()
            df = df_1.copy()
            #extract only the columns with the parameters we want to use in the model, if index_parameters is not None
            if index_parameters is not None:
                df_params = df[index_parameters]
                df = pd.concat([df.iloc[:, :7], df_params], axis=1)
             # PREPROCESS
            pre = BehaviorPreprocessor()
            (behavior_df, X_scaled, metadata_df) = pre.preprocess(df)
            Add_Glicko = behavior_df["Last.day.Glicko"] 
            # PCA

            pca = PersonalityPCA(self.config.n_pca_components)
            #optimize PCA and clusters
            pca_results = pca.optimize_pca_and_clusters(X_scaled,  pca_range=range(2, 16),
                        cluster_range=range(2, 7),random_state=42)
            
            pca.plot_pca_clusters(X_scaled, 2, 3, random_state=42, output_dir = output_dir, meta_data = metadata_df, glicko = None, add_hierarchy = True)

            ##############
            pca = PersonalityPCA(n_components = 2)
            X_pca = pca.fit_transform(X_scaled)
            print("\nExplained variance:")
            print(pca.explained_variance())

            # Save PCA results to Excel
            pca_results.to_excel(f"{output_dir}/PCA_Optimization_Results.xlsx", index=False)
            print(f"\nSaved PCA results to: {output_dir}/PCA_Optimization_Results.xlsx")

            # ====================================================
            # MODEL SELECTION
            # ====================================================
            evaluator = ModelEvaluator()
            selection = evaluator.find_optimal_archetypes(X_pca,k_range=range(2, 7))
            best_k = selection["best_k"]
           # self.config.n_archetypes = best_k
            print(f"\nSelected archetypes: {best_k}")

            # ARCHETYPES
            archetypes = ArchetypeModel(n_archetypes=3,random_state=self.config.random_state)
            archetypes.fit(X_pca)
            labels = archetypes.predict(X_pca)
            probabilities = archetypes.predict_proba(X_pca)

            #Model quality metrics
            bic = archetypes.bic(X_pca)
            aic = archetypes.aic(X_pca)
            print("\nModel quality:")
            print(f"BIC: {bic:.2f}")
            print(f"AIC: {aic:.2f}")

            # RESULTS
            builder = ResultsBuilder()
            results = builder.build_results_table(metadata_df,  labels, probabilities,  X_pca)

             # CORRELATIONS
            corr = ArchetypeCorrelationAnalyzer()
            correlation_df = corr.compute_correlations(behavior_df, probabilities)
            top_features = corr.top_features(correlation_df)
            corr.plot_significant_correlations(correlation_df, output_dir = output_dir)

            # VISUALIZATION
            viz = PersonalityVisualizer()
            
            if X_pca.shape[1] == 2:
               fig1 = viz.plot_2d_space(X_pca, labels, archetypes.centers())
               fig1.savefig(f"{output_dir}/2d_archetype_space.pdf", format="pdf", dpi=600, bbox_inches="tight")
            elif X_pca.shape[1] == 3:   
                 fig1=viz.plot_3d_space(X_pca, labels, archetypes.centers())
                 fig1.savefig(f"{output_dir}/3d_archetype_space.pdf", format="pdf", dpi=600, bbox_inches="tight")
            
            fig2=viz.plot_archetype_counts(labels)
            fig2.savefig(f"{output_dir}/archetype_counts.pdf", format="pdf", dpi=600, bbox_inches="tight")

            plt.close(fig1)
            plt.close(fig2)

            return {
            "behavior_df": behavior_df,
            "X_scaled": X_scaled,
            "X_pca": X_pca,
            "labels": labels,
            "probabilities": probabilities,
            "results": results,
            "correlations": correlation_df,
            "top_features": top_features,
            "bic_score": bic,
             "aic_score": aic,
             "behavior_parameters_used": pd.Series(behavior_df.columns)
              }


    #====================================================
    #Save to excel
    #====================================================

    def save_results_to_excel(self, output_file, behavior_results):

        with pd.ExcelWriter(
            output_file,
            engine="openpyxl"
        ) as writer:

            # ====================================================
            # RESULTS TABLE
            # ====================================================

            behavior_results["results"].to_excel(
                writer,
                sheet_name="Assignments",
                index=False
            )

            # ====================================================
            # CORRELATIONS
            # ====================================================

            behavior_results["correlations"].to_excel(
                writer,
                sheet_name="Correlations",
                index=False
            )

            # ====================================================
            # PCA SPACE
            # ====================================================

            n_pcs = behavior_results["X_pca"].shape[1]

            pca_df = pd.DataFrame(
                  behavior_results["X_pca"],
                  columns=[
                     f"PC{i+1}"
                     for i in range(n_pcs)
                       ]
                       )

            pca_df.to_excel(
                writer,
                sheet_name="PCA",
                index=False
            )

            # ====================================================
            # ARCHETYPE PROBABILITIES
            # ====================================================

            n_archetypes = behavior_results["probabilities"].shape[1]

            prob_df = pd.DataFrame(
                behavior_results["probabilities"],
                columns=[
                    f"Probability_A{i}"
                    for i in range(n_archetypes)
                ])

            prob_df.to_excel(
                writer,
                sheet_name="Probabilities",
                index=False
            )

            # ====================================================
            # TOP FEATURES
            # ====================================================

            behavior_results["top_features"].to_excel(
                writer,
                sheet_name="Significant_correlations",
                index=False
            )
            #===========================================
            # all behaviour parameters used in the model
            #===========================================
            behavior_results["behavior_parameters_used"].to_excel(
                writer,
                sheet_name="Behavior_Parameters",
                index=False
            )






        print(
            f"\nSaved results to: {output_file}"
        )

    #=================================================
    # Hormone join with behaviour
    #==========================================================
    def join_hormone_behavior(self, hormones_data_file, output_dir):
        hormone_df = pd.read_excel(f"{hormones_data_file}") 
        behaviour_results = pd.read_excel(f"{output_dir}/Archetype_Results.xlsx", sheet_name="Assignments") 

        # Merge behavior results with hormone data
        # keep original dataframe
        hormone_X = hormone_df.copy()

        # normalize ONLY columns 8 onward
        #hormone_X.iloc[:, 8:] = StandardScaler().fit_transform(hormone_X.iloc[:, 8:])

        #find common columns to merge on
        common_cols = sorted(
            set(behaviour_results.columns).intersection(hormone_X.columns)
        )
        print("Common columns used for merge:")
        print(common_cols)

        wanted_cols = ["Assigned_Archetype", "Probability_A0", "Probability_A1", "Probability_A2"] #ADJUST
        selected_cols = common_cols + [col for col in wanted_cols if col not in common_cols]
        merged_df = pd.merge(
            hormone_X,
            behaviour_results[selected_cols],
            on=common_cols,
            how="inner"
        )

        # Save merged data for hormone prediction
        merged_df.to_excel(f"{output_dir}/Merged_Hormone_Behavior_Data.xlsx", index=False)
        print(f"\nSaved merged data to: {output_dir}/Merged_Hormone_Behavior_Data.xlsx")
        return merged_df
    #===========================================================
    # Hormone prediction
    #===========================================================
    def run_hormone_prediction(self, hormone_X, archetype_probabilities, output_dir):
        predictor = HormonePredictionModel()
        Y_pred = predictor.cross_validated_prediction(hormone_X,archetype_probabilities)
        archetype_probabilities = archetype_probabilities.values
        
        r2_scores = predictor.compute_r2(archetype_probabilities,Y_pred)
        print("\nCross-validated R²:")
        for i, r2 in enumerate(r2_scores):
            print(f"A{i}: {r2:.3f}")
        predictor.plot_predictions(archetype_probabilities, Y_pred, archetype_names=None, save_path = output_dir) 
        baseline_r2 = predictor.compute_baseline_r2(archetype_probabilities)
        print("\nBaseline R²:")
        for i, r2 in enumerate(baseline_r2):
            print(f"A{i}: {r2:.3f}")

        #  # PERMUTATION
        # perm = PermutationTester(self.config.permutation_iterations)
        # perm_r2, p_values = perm.run(predictor, hormone_X, archetype_probabilities, r2_scores)
        # print("\nPermutation test p-values:")
        # for i, p in enumerate(p_values):
        #     print(f"A{i}: {p:.3f}")
        # perm.plot_histograms({"perm_r2": perm_r2, "true_r2": r2_scores, "p_values": p_values}, archetype_names=None, save_file=f"{output_dir}/Permutation_Histograms.tiff")

        # return {
        #     "predictions": Y_pred,
        #     "r2_scores": r2_scores,
        #     "baseline_r2": baseline_r2,
        #     "perm_r2": perm_r2,
        #     "p_values": p_values
        # }

        ###################################################
        #######Hormone prediction  with 4 classes and 3 pcs##############
    # ============================================================
    # CONFIGURATION CLASS
    # ============================================================

    def load_index_parameters(self, auxiliary_file):
        # Load workbook
        wb = load_workbook(auxiliary_file)
        ws = wb["Behavior_Parameters"]

        yellow_rows = []
        parameters = []

        for cell in ws["A"]:
            if cell.fill.patternType == "solid":
                color = cell.fill.fgColor.rgb
                if color in ["FFFFFF00", "FFFF00", "00FFFF00"]:
                    yellow_rows.append(cell.row)
                    name = ws[f"A{cell.row}"].value
                    parameters.append(name)

        print(yellow_rows)
        print(parameters)
        return parameters