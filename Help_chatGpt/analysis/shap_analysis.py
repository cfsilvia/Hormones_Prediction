import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.stats import mannwhitneyu
import numpy as np
import seaborn as sns
from statsmodels.stats.multitest import multipletests

class   ShapAnalysis:
    def __init__(self, model, X, feature_names, class_names, output_dir, meta_data=None):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = output_dir
        self.meta_data = meta_data

    def run(self):
         self.shap_dir = os.path.join(self.output_dir,"shap_outputs")
         os.makedirs(self.shap_dir, exist_ok=True)
         #transform the data - create pipeline with only transformer steps (imputer and scaler)
         transformers = Pipeline(self.model.steps[:-1])
         X_processed = transformers.transform(self.X)
         classifier = self.model.named_steps["clf"]
         explainer = shap.Explainer( classifier.predict_proba, X_processed)
         shap_values = explainer(X_processed)

         print("\nSHAP shape:")

         print(shap_values.values.shape)

         # ====================================================
         # SAVE SHAP VALUES TO EXCEL
         # ====================================================
         shap_excel_path = os.path.join(self.shap_dir, "shap_values.xlsx")
         with pd.ExcelWriter(shap_excel_path, engine="openpyxl") as writer:
             shap_feature_names = [f"shap_{name}" for name in self.feature_names]
             X_df = pd.DataFrame(self.X, columns=self.feature_names) if not isinstance(self.X, pd.DataFrame) else self.X.reset_index(drop=True)
             meta_df = None
             if self.meta_data is not None:
                 meta_df = self.meta_data.reset_index(drop=True) if isinstance(self.meta_data, pd.DataFrame) else pd.DataFrame(self.meta_data)
             for class_idx in range(len(self.class_names)):
                 class_shap = pd.DataFrame(shap_values.values[:, :, class_idx], columns=shap_feature_names)

                 if meta_df is not None and len(meta_df) == len(X_df):
                     combined = pd.concat([meta_df, X_df.reset_index(drop=True), class_shap], axis=1)
                 elif len(X_df) == class_shap.shape[0]:
                     combined = pd.concat([X_df.reset_index(drop=True), class_shap], axis=1)
                 else:
                     combined = class_shap

                 combined.to_excel(writer, sheet_name=f"Class_{self.class_names[class_idx]}", index=False)
         print(f"\nSaved SHAP values to: {shap_excel_path}")

          # ====================================================
        # GLOBAL SHAP IMPORTANCE
        # ====================================================

         mean_abs_shap = np.mean(np.abs(shap_values.values),axis=(0, 2))

         shap_df = pd.DataFrame({"Feature": self.feature_names,"Importance": mean_abs_shap })

         shap_df = shap_df.sort_values(by="Importance", ascending=False)

         print("\nTop SHAP Features:")

         print(shap_df.head(20))
        
          # ====================================================
        # CLASS-SPECIFIC SHAP IMPORTANCE
        # ====================================================

         print("\nGenerating class-specific SHAP plots...")

         n_classes = len(self.class_names)

         fig, axes = plt.subplots(1,n_classes, figsize=(7 * n_classes, 8))

         # store all class dfs
         class_shap_results = []
         for class_idx in range(n_classes):
             #mean abs for each class
             class_importance = np.mean(np.abs(shap_values.values[ :, :,class_idx]), axis=0)
             class_df = pd.DataFrame({

                "Feature": self.feature_names,

                "Importance": class_importance
            })

             class_df = class_df.sort_values(by="Importance", ascending=False)

             class_shap_results.append(class_df)

             # ================================================
            # BARPLOT
            # ================================================

             axes[class_idx].barh(class_df["Feature"][::-1], class_df["Importance"][::-1])

             axes[class_idx].set_title(f"Class: " f"{self.class_names[class_idx]}")
             axes[class_idx].set_xlabel("Mean |SHAP|")
             axes[class_idx].set_xlim(0, 0.07)

         plt.tight_layout()
         plt.savefig(os.path.join(self.shap_dir, "class_specific_shap_importance.pdf"), dpi=300, bbox_inches="tight")

         plt.close()
        
       # ====================================================
       # COMBINED SHAP VIOLIN PLOTS
       # SORTED SEPARATELY PER CLASS
       # ====================================================

         print("\nGenerating class-specific sorted violin plots...")

         


         for class_idx in range(n_classes):

            
            class_importance = np.mean(np.abs(shap_values.values[:, :, class_idx]), axis=0)

            sorted_idx = np.argsort(class_importance)[::-1]

            ordered_feature_names = [self.feature_names[i] for i in sorted_idx]

            # reorder SHAP
            shap_sorted = shap_values.values[:, sorted_idx, class_idx]

            # reorder X
            X_sorted = X_processed[ :, sorted_idx]

        # ================================================
        # VIOLIN PLOT
        # ================================================
            plt.figure(figsize=(9, 14))

            shap.summary_plot(shap_sorted, X_sorted, feature_names=ordered_feature_names, plot_type="violin",
                         max_display=len(ordered_feature_names), sort=False, show=False)
            axes[class_idx].set_title(f"{self.class_names[class_idx]}", fontsize=14, weight="bold")

            plt.title(f"SHAP Violin Plot\n" f"Class: " f"{self.class_names[class_idx]}", fontsize=16, weight="bold")

            plt.tight_layout()
            
            plt.savefig(os.path.join(self.shap_dir, f"violin_" f"{self.class_names[class_idx]}" f".pdf"),dpi=300,
               bbox_inches="tight")
            plt.close()

         

         print("\nClass-specific violin plots saved." )


    def plot_shap_violin_by_sex(self, combined_file, output_dir,sex_col="sex",shap_prefix="shap_",top_n=None):
         os.makedirs(output_dir, exist_ok=True)
         xls = pd.ExcelFile(combined_file)
         all_pvalues = {}

         for sheet in xls.sheet_names:
            print(f"\nProcessing class: {sheet}")
            df = pd.read_excel(combined_file, sheet_name=sheet)
            shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
            df[sex_col] = (df[sex_col].astype(str).str.lower().str.strip())
            male_df = df[df[sex_col] == "male"].copy()
            female_df = df[df[sex_col] == "female"].copy()
        # --------------------------------------------------
        # Order features by male importance
        # --------------------------------------------------
            male_importance = (df[shap_cols].abs().mean().sort_values(ascending=False))
            ordered_shap_cols = male_importance.index.tolist()
            ordered_features = [c.replace(shap_prefix, "") for c in ordered_shap_cols]
            # for statistics
            pval_rows = []

            for shap_col, feature in zip(ordered_shap_cols, ordered_features):
                male_vals = male_df[shap_col].dropna()
                female_vals = female_df[shap_col].dropna()

                if len(male_vals) > 0 and len(female_vals) > 0:
                   stat, pval = mannwhitneyu(male_vals,female_vals,alternative="two-sided")

                   rank_biserial = (
                    (2 * stat)
                    / (len(male_vals) * len(female_vals))
                ) - 1
                else:
                  stat = np.nan
                  pval = np.nan
                  rank_biserial = np.nan

                pval_rows.append({
                "Class": sheet,
                "Feature": feature,
                "Male_n": len(male_vals),
                "Female_n": len(female_vals),
                "Male_mean_abs_SHAP": male_vals.abs().mean(),
                "Female_mean_abs_SHAP": female_vals.abs().mean(),
                "MannWhitney_U": stat,
                "Rank_biserial": rank_biserial,
                "pvalue": pval
            })

            pval_df = pd.DataFrame(pval_rows)

            # Apply Benjamini-Hochberg correction to p-values
            if len(pval_df) > 0 and not pval_df["pvalue"].isna().all():
                reject, pvals_corrected, _, _ = multipletests(pval_df["pvalue"].fillna(1.0), method="fdr_bh")
                pval_df["pvalue_BH"] = pvals_corrected
                pval_df["significant_BH"] = reject
            else:
                pval_df["pvalue_BH"] = pval_df["pvalue"]
                pval_df["significant_BH"] = False

            all_pvalues[sheet] = pval_df
            ##
            
            # --------------------------------------------------
            # Build plotting dataframe
            # --------------------------------------------------

        # --------------------------------------------------
            # SHAP summary violin plots
            # --------------------------------------------------

            # Matching original feature-value columns
            feature_cols = ordered_features

            

            male_shap = male_df[ordered_shap_cols].values
            female_shap = female_df[ordered_shap_cols].values

            male_features = male_df[feature_cols].values
            female_features = female_df[feature_cols].values

            # Shared x-axis limits
            xmin = np.nanmin([male_shap.min(), female_shap.min()])
            xmax = np.nanmax([male_shap.max(), female_shap.max()])
            pad = 0.08 * (xmax - xmin)
            xlim = (xmin - pad, xmax + pad)

            
            fig, axes = plt.subplots(
                1, 2,
                figsize=(15, max(6, len(ordered_features) * 0.35)),
                sharex=False,
                sharey=True,
                gridspec_kw={"wspace": 0.55}
            )

            # -------------------------
            # Male plot
            # -------------------------
            plt.sca(axes[0])
            shap.summary_plot(
                male_shap,
                features=male_features,
                feature_names=ordered_features,
                plot_type="violin",
                max_display=len(ordered_features),
                sort=False,
                show=False,
                color_bar=False
            )
            axes[0].set_title("Male")
            axes[0].set_xlim(xlim)

            # keep only left colorbar
            #ig_axes_after_male = fig.axes.copy()

            # -------------------------
            # Female plot
            # -------------------------
            plt.sca(axes[1])
            shap.summary_plot(
                female_shap,
                features=female_features,
                feature_names=ordered_features,
                plot_type="violin",
                max_display=len(ordered_features),
                sort=False,
                show=False,
                color_bar=True
            )
            axes[1].set_title("Female")
            axes[1].set_xlim(xlim)

            # Force same y labels/order as male
            axes[1].set_yticks(axes[0].get_yticks())
            axes[1].set_yticklabels(ordered_features[::-1])

            # -------------------------
            # Significance symbols in middle (aligned with y-labels)
            # -------------------------
            tick_texts = [t.get_text() for t in axes[0].get_yticklabels()]
            ticks = axes[0].get_yticks()
            for _, row in pval_df.iterrows():
                feature = row["Feature"]
                p = row["pvalue"]

                if pd.isna(p):
                    continue
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                elif p < 0.1:
                    sig = "#"
                else:
                    sig = ""

                # compute figure y coordinate for the feature label
                y_fig = None
                if tick_texts and any(tick_texts):
                    try:
                        idx = tick_texts.index(feature)
                        y_data = ticks[idx]
                        y_disp = axes[0].transData.transform((0, y_data))[1]
                        y_fig = fig.transFigure.inverted().transform((0, y_disp))[1]
                    except ValueError:
                        y_fig = None

                if y_fig is None:
                    # fallback: compute relative position based on ordered_features
                    feature_idx = ordered_features.index(feature)
                    y_fig = (
                        axes[0].get_position().y0
                        + (len(ordered_features) - feature_idx - 0.5)
                        / len(ordered_features)
                        * axes[0].get_position().height
                    )

                fig.text(0.515, y_fig, sig, ha="center", va="center", fontsize=10)

            plt.suptitle(f"SHAP Summary Plot by Sex\n{sheet}", y=1.02, fontsize=14)
            plt.tight_layout()

            plot_path = os.path.join(self.shap_dir, f"shap_summary_by_sex_{sheet}.pdf")
            plt.savefig(plot_path, dpi=600, bbox_inches="tight")
            plt.close()

            # After processing all sheets, save concatenated p-values to the SHAP Excel file
            if all_pvalues:
                combined_pvals = pd.concat(all_pvalues.values(), ignore_index=True)
                shap_dir = os.path.join(output_dir, "shap_outputs")
                os.makedirs(shap_dir, exist_ok=True)
                shap_excel_path = os.path.join(shap_dir, "shap_values.xlsx")
                if os.path.exists(shap_excel_path):
                    with pd.ExcelWriter(shap_excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                        combined_pvals.to_excel(writer, sheet_name="pvalue", index=False)
                else:
                    with pd.ExcelWriter(shap_excel_path, engine="openpyxl") as writer:
                        combined_pvals.to_excel(writer, sheet_name="pvalue", index=False)
                print(f"\nSaved p-values to: {shap_excel_path} (sheet 'pvalue')")