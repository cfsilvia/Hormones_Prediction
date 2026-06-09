import pandas as pd
import os
#from utils.data_loader import HormoneDataLoader

from analysis import shap_analysis
from utils.helpers import confidence_interval

from models.gaussian_nb_model import GaussianNBPipeline

from models.nested_cv import NestedCrossValidator

from analysis.metrics import MetricsAnalyzer

from analysis.visualization import VisualizationTools

from sklearn.preprocessing import LabelEncoder
from analysis.permutation_test import PermutationTest
from analysis.shap_analysis import ShapAnalysis
import matplotlib

import matplotlib.pyplot as plt




class PersonalityPipelineMulticlass:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def run_prediction_multiclass(self, X, y, output_dir,meta_data=None):
         #y = y.iloc[:, 0]
         # Label encoding
         y_encoded = self.label_encoder.fit_transform(y)
         

         print("=" * 60)

         print("X shape:", X.shape)
         print("y shape:", y.shape)

         print("\nLabel distribution:")
         print(pd.Series(y).value_counts())

         print("\nClasses:")
         print(self.label_encoder.classes_)

         print("=" * 60)
         # ============================================================
         # MODEL
         # ============================================================
         model = GaussianNBPipeline()
         # ============================================================
         # NESTED CV
         # ============================================================
         nested_cv = NestedCrossValidator(model)
         results = nested_cv.run(X, y_encoded)

         # ============================================================
         # CONFIDENCE INTERVALS
         # ============================================================
         bal_mean, bal_low, bal_high = (confidence_interval(results["bal_scores"]))
         f1_mean, f1_low, f1_high = (confidence_interval(results["f1_scores"]))
         print("\nBalanced Accuracy")
         print(f"{bal_mean:.3f} " f"[{bal_low:.3f}, {bal_high:.3f}]")
         print("\nMacro F1")
         print(f"{f1_mean:.3f} " f"[{f1_low:.3f}, {f1_high:.3f}]")
         # ============================================================
         # METRICS
         # ============================================================
         metrics = MetricsAnalyzer(results["y_true"], results["y_pred"], self.label_encoder.classes_)
         report_metrics = metrics.classification_report()
        
         cm, cm_df = metrics.confusion_matrix()
         #save
         excel_path =  os.path.join(output_dir, "classification_results.xlsx")
         with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
              # Classification report
                report_metrics.to_excel(writer, sheet_name="Classification_Report")

                # Raw confusion matrix
                cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
                print(f"\nSaved Excel results to: {excel_path}")


         # ============================================================
         # VISUALIZATION
         # ============================================================
         VisualizationTools.plot_confusion_matrix(cm, self.label_encoder.classes_, output_dir)  
         
         # ============================================================
         # PERMUTATION TEST
         # ============================================================

         print("\n" + "=" * 60)

         print("Permutation ANALYSIS")

         print("=" * 60)
         perm = PermutationTest(results["best_model"], X, y)
         results_p = perm.run_permutation_test()
         #permutation of each class vs all others
         results_p_per_class = perm.run_permutation_test_per_class(self.label_encoder.classes_)


         #histogram permutation
         VisualizationTools.plot_permutation(results_p["permutation_scores"], results_p["score"], results_p["pvalue"], output_dir)
         #histogram permutation per class
         VisualizationTools.plot_permutation_per_class(results_p_per_class, output_dir)
         #============================================================
         # SHAP ANALYSIS
         # ============================================================

         print("\n" + "=" * 60)

         print("SHAP ANALYSIS")

         print("=" * 60)

         shap = ShapAnalysis(results["best_model"], X, X.columns.tolist(), self.label_encoder.classes_, output_dir, meta_data=meta_data)

         shap.run()
         combined_file = os.path.join(output_dir, "shap_outputs", "shap_values.xlsx")

         shap.plot_shap_violin_by_sex(combined_file, output_dir,sex_col="sex",shap_prefix="shap_",top_n=None)
