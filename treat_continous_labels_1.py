import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import pearsonr

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import os
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.special import rel_entr
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor



class treat_continous_labels:

    def __init__(self, data, output_dir, run_features_normalization, run_model, sex = None):
        self.data = data
        self.output_dir = output_dir
        self.sex = sex
        self.run_features_normalization = run_features_normalization
        self.run_model = run_model

        
    def __call__(self):

        X, y, feature_names, metadata = self.load_data()


        y_true, y_pred, rmse, cosine_sim, kl_div, ent = self.loocv_pipeline(X, y)
        self.confusion_matrix(y_true, y_pred)
        self.save_predictions(y_true, y_pred)
        self.plot_prediction_score(y_true, y_pred)  
        self.rsquare( y_true, y_pred)
        self.evaluate_probabilistic_predictions(y_true, y_pred) 
        self.scatter_plot(y_true, y_pred)

        (
            original_rmse, permuted_rmses, p_value_rmse, 
            original_cosine,permutated_cosines, p_value_cosine,
            original_kl,permutated_kls, p_value_kl,
            original_ent,permutated_ents, p_value_ent) = self.permutation_test(X, y)

        self.plot_permutation_test_results(permuted_rmses, original_rmse, p_value_rmse)
        self.plot_permutation_test_results_cosine(permutated_cosines, original_cosine, p_value_cosine)
        self.plot_permutation_test_results_kl(permutated_kls, original_kl, p_value_kl) 
        self.plot_permutation_test_results_entropy(permutated_ents, original_ent, p_value_ent)

        shap_results, X_scaled_df =self.shap_values(X, y, feature_names, metadata)
        self.save_shap_values(shap_results, feature_names)
        self.bar_map_plot(shap_results, feature_names)
        self.summary_plot(shap_results, X_scaled_df, feature_names)
        
        self.plot_violin_shap_values(X_scaled_df, feature_names, metadata)






    ################ LOAD DATA ################

    def load_data(self):

        data = pd.read_excel(self.data)
        #sex selection
        if self.sex is not None:
            data = data[data['sex'] == self.sex]


        metadata_cols = ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips','Last.day.Glicko','Animal']

        metadata = data[metadata_cols]

        if self.sex  is not None:
           X = data.drop(
            ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips',
             'Last.day.Glicko','Animal', 'sexFeature','Arch1','Arch2','Arch3','Arch4'], axis=1)
        else: 
           X = data.drop(
            ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips',
             'Last.day.Glicko','Animal','sexFeature', 'Arch1','Arch2','Arch3','Arch4'], axis=1)

        y = data[['Arch1','Arch2','Arch3','Arch4']]

        feature_names = X.columns.tolist()

        return X, y, feature_names, metadata



    ################ MODEL ################

    def build_model(self):
        #alpha 0.5 1
        models = {'linear' :  MultiOutputRegressor(Ridge(alpha= 1)), 'tree': MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)) }
        
     
        # alphas = np.logspace(-3, 3, 50)

        # model = MultiOutputRegressor(RidgeCV(alphas=alphas))
       # model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200,  max_depth=None, max_features="sqrt",  min_samples_leaf=2, n_jobs=-1, random_state=42))
        #model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42))
        
        
       # model = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
       
        return models[self.run_model]





    ################ LOOCV ################

    def loocv_pipeline(self, X, y):

        loo = LeaveOneOut()

        y_true = []
        y_pred = []
        
        X = np.asarray(X)
        y = np.asarray(y)
     
        index = 0
        # alphas_used = []

        for train_idx, test_idx in loo.split(X):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            #Normalize features
            if self.run_features_normalization:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            
            # preds_list = []
            # for i in range(4):
            #     model = self.build_model()
            #     model.fit(X_train, y_train[:,i])
            #     pred_i = model.predict(X_test)
            #     preds_list.append(pred_i)
            
            # preds = np.column_stack(preds_list)
            #Train model
            model = self.build_model()
            model.fit(X_train, y_train)

            #alpha taken
            
            # for i, est in enumerate(model.estimators_):
            #     alphas_used.append(est.alpha_)
            ###########

            preds = model.predict(X_test)
           


             #preds = softmax(preds, axis=1)
            # ensure positive
            preds = np.clip(preds, 0, None)
            # normalize to sum to 1
            preds = preds / preds.sum(axis=1, keepdims=True)

            y_true.append(y_test[0])
            y_pred.append(preds[0])

            index += 1
           # print(f"LOOCV Progress: {index}/{len(X)}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        row_sums_true = np.sum(y_true, axis=1)
        row_sums_pred = np.sum(y_pred, axis=1)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"RMSE: {rmse}")
        cosine_sim, kl_div, ent = self.evaluate_probabilistic_predictions(y_true, y_pred, plot = False)
        #print("Mean alpha:", np.mean(alphas_used))


        return y_true, y_pred, rmse, cosine_sim, kl_div, ent

    
    ##################calculate r square###################
    def rsquare(self, y_true, y_pred):
        archetypes = ["Arch1","Arch2","Arch3","Arch4"]
        r2_scores = []
        for i, arc in enumerate(archetypes):
            r2 = r2_score(y_true[:,i], y_pred[:,i])
            r2_scores.append(r2)
            print(f"{arc} R2 = {r2:.3f}")

        plt.figure(figsize=(6,4))

        plt.bar(archetypes, r2_scores)

        plt.axhline(0, linestyle="--")  # reference line

        plt.ylabel("R² score")
        plt.title("Cross-validated R² per Archetype")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'r2_scores.pdf'))
        plt.close() 

        return r2_scores


    ################ Permutation test ################

    def permutation_test(self, X, y, n_permutations=200):

        original_rmse, original_cosine, original_kl, original_ent= self.loocv_pipeline(X, y)[2:6]

        permuted_rmses = []
        permutated_cosines = []
        permutated_kls = []
        permutated_ents = []    

        for i in range(n_permutations):
            y_permuted = np.random.permutation(y)
            permuted_rmse, cosine_sim, kl_div, ent= self.loocv_pipeline(X, y_permuted)[2:6]
            permuted_rmses.append(permuted_rmse)
            permutated_cosines.append(cosine_sim)
            permutated_kls.append(kl_div)
            permutated_ents.append(ent)

            print(f"Permutation Test Progress: {i+1}/{n_permutations}")


        p_value_rmse = np.mean(permuted_rmses <= original_rmse)
        p_value_cosine = np.mean(permutated_cosines >= original_cosine)
        p_value_kl = np.mean(permutated_kls <= original_kl)
        p_value_ent = np.mean(permutated_ents <= original_ent)

        return (
            original_rmse, permuted_rmses, p_value_rmse, 
        original_cosine,permutated_cosines, p_value_cosine,
        original_kl,permutated_kls, p_value_kl,
        original_ent,permutated_ents, p_value_ent
        )

    
    #Plotting function for permutation test results
    def plot_permutation_test_results(self, permuted_rmses, original_rmse, p_value_rmse):

        plt.hist(permuted_rmses, bins=20, alpha=0.7, color='blue', label='Permuted RMSEs')
        plt.axvline(original_rmse, color='red', linestyle='dashed', linewidth=2, label=f'Original RMSE: {original_rmse:.4f}')
        if self.sex is not None:
            plt.title(f'Permutation Test Results for {self.sex}\np-value: {p_value_rmse:.4f}')
        else:
            plt.title(f'Permutation Test Results\np-value: {p_value_rmse :.4f}')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.legend()
        if self.sex is not None:
            plt.savefig(self.output_dir + 'permutation_test_results_' + self.sex + '.pdf')
        else:
            plt.savefig(self.output_dir + 'permutation_test_results.pdf')
       # plt.show()
    

     #Plotting function for permutation test results cosine
    def plot_permutation_test_results_cosine(self, permutated_cosines, original_cosine, p_value_cosine):
        plt.figure()
        plt.hist(permutated_cosines, bins=20, alpha=0.7, color='blue', label='Permuted cosine similarity')
        plt.axvline(original_cosine, color='red', linestyle='dashed', linewidth=2, label=f'Original cosine: {original_cosine:.4f}')
        if self.sex is not None:
            plt.title(f'Permutation Test Results for {self.sex}\np-value: {p_value_cosine:.4f}')
        else:
            plt.title(f'Permutation Test Results\np-value: {p_value_cosine:.4f}')
        plt.xlabel('cosine similarity')
        plt.ylabel('Frequency')
        plt.legend()
        if self.sex is not None:
            plt.savefig(self.output_dir + 'permutation_test_results_cosine_' + self.sex + '.pdf')
        else:
            plt.savefig(self.output_dir + 'permutation_test_results_cosine_.pdf')


     #Plotting function for permutation test results
    def plot_permutation_test_results_kl(self, permutated_kls, original_kl, p_value_kl):
        plt.figure()
        plt.hist(permutated_kls, bins=20, alpha=0.7, color='blue', label='Permuted kl divergence')
        plt.axvline(original_kl, color='red', linestyle='dashed', linewidth=2, label=f'Original kl: {original_kl:.4f}')
        if self.sex is not None:
            plt.title(f'Permutation Test Results for {self.sex}\np-value: {p_value_kl:.4f}')
        else:
            plt.title(f'Permutation Test Results\np-value: {p_value_kl:.4f}')
        plt.xlabel('kl divergence')
        plt.ylabel('Frequency')
        plt.legend()
        if self.sex is not None:
            plt.savefig(self.output_dir + 'permutation_test_results_kl_' + self.sex + '.pdf')
        else:
            plt.savefig(self.output_dir + 'permutation_test_results_kl_.pdf')


    #Plotting function for permutation test results
    def plot_permutation_test_results_entropy(self, permutated_ents, original_ent, p_value_ent):
        plt.figure()
        plt.hist(permutated_ents, bins=20, alpha=0.7, color='blue', label='Permuted entropy')
        plt.axvline(original_ent, color='red', linestyle='dashed', linewidth=2, label=f'Original entropy: {original_ent:.4f}')
        if self.sex is not None:
            plt.title(f'Permutation Test Results for {self.sex}\np-value: {p_value_ent:.4f}')
        else:
            plt.title(f'Permutation Test Results\np-value: {p_value_ent:.4f}')
        plt.xlabel('entropy')
        plt.ylabel('Frequency')
        plt.legend()
        if self.sex is not None:
            plt.savefig(self.output_dir + 'permutation_test_results_entropy_' + self.sex + '.pdf')
        else:
            plt.savefig(self.output_dir + 'permutation_test_results_entropy_.pdf')
    

    ################ SHAP on the full dataset ################

    def shap_values(self, X, y, feature_names, metadata):
       
        print("\nComputing SHAP explanations\n")
        archetypes = ["Arch1","Arch2","Arch3","Arch4"]

        shap_results = {}


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = self.build_model()
        model.fit(X_scaled, y)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names) 

        filepath = os.path.join(self.output_dir, 'shap_values.xlsx')
        with pd.ExcelWriter(filepath) as writer:
          for i, arc in enumerate(archetypes):
            explainer = shap.Explainer(model.estimators_[i], X_scaled_df)
            shap_values = explainer(X_scaled_df)
            shap_results[arc] = shap_values

            shap_vals = shap_results[arc].values
            df = pd.DataFrame(shap_vals, columns=feature_names)
            df = pd.concat([metadata, df], axis=1)
            df.to_excel(writer, sheet_name=arc, index=False)

        return shap_results, X_scaled

#############Save shap values importance in excel file#############
    def save_shap_values(self, shap_results, feature_names):
        filepath = os.path.join(self.output_dir, 'shap_values_importance.xlsx')
        with pd.ExcelWriter(filepath) as writer:
            for arc, shap_values in shap_results.items():
                shap_importance = np.abs(shap_values.values).mean(axis=0)
                importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
                importance_df.sort_values(by='SHAP Importance', ascending=False, inplace=True)
                importance_df.to_excel(writer, sheet_name=arc + '_importance', index=False)

###############bar map#############
    def bar_map_plot(self, shap_results, feature_names):
        
        # Compute mean(|SHAP|) for each archetype and create a DataFrame
        importance_df = pd.DataFrame(
            [np.abs(shap_values.values).mean(axis=0) for shap_values in shap_results.values()],
            index=shap_results.keys(),
            columns=feature_names
        )

        # Optional: show only top features overall
        #top_features = importance_df.mean(axis=0).sort_values(ascending=False).head(20).index
        #importance_df = importance_df[top_features]
        # Plot bar
        importance_df.T.plot(kind="bar",figsize=(12,6))

        plt.ylabel("Mean |SHAP|")
        plt.title("Feature Importance across Archetypes")
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'shap_bar.pdf'))
        plt.close()

######################summary plot###############
    def summary_plot(self, shap_results, X_scaled_df, feature_names):
        n_features = X_scaled_df.shape[1]
        X_scaled_df = pd.DataFrame(X_scaled_df, columns=feature_names)
        fig, axes = plt.subplots(2,2, figsize=(20,10))
        axes = axes.flatten()

        for i, (arc, shap_values) in enumerate(shap_results.items()):
            plt.sca(axes[i])
            axes[i].set_axis_on()
            shap.summary_plot(shap_values.values, X_scaled_df, plot_type="bar",show=False, max_display=n_features )  
            axes[i].tick_params(axis='y', labelsize=9)
            axes[i].tick_params(axis='x', labelsize=8)
            # reduce the x-axis label font
            axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=8)
            axes[i].set_title(arc)

        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'summary_plot.pdf'), dpi = 300)
        plt.close()

########scatter plot ###########
    def scatter_plot(self,y_true, y_pred):
        archetypes = ["Arch1","Arch2","Arch3","Arch4"]

        fig, axes = plt.subplots(2,2, figsize=(10,10))
        axes = axes.flatten()

        for i, arc in enumerate(archetypes):

            r, p = pearsonr(y_true[:,i], y_pred[:,i])

            axes[i].scatter(y_true[:,i], y_pred[:,i], alpha=0.7)

            axes[i].plot([0,1],[0,1], 'r--')  # ideal prediction line

            axes[i].set_xlabel("True")
            axes[i].set_ylabel("Predicted")

            axes[i].set_title(f"{arc}\n r = {r:.2f}, p = {p:.6f}")

        plt.tight_layout()
        if self.sex is not None:
            plt.savefig(os.path.join(self.output_dir, self.sex + '_correlation_plot.pdf'), dpi = 300)
        else:
            plt.savefig(os.path.join(self.output_dir, 'correlation_plot.pdf'), dpi = 300)
        plt.close()

##########################plot for each archetype a plot violin#########################
    def plot_violin_shap_values(self, X_scaled_df, feature_names, metadata):
        output_excel = os.path.join(self.output_dir, "SHAP_sex_differences.xlsx")
        writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")
        #read all sheets
        sheets = pd.read_excel(os.path.join(self.output_dir + 'shap_values.xlsx'), sheet_name=None)

        # Metadata columns
        meta_cols = ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips', 'Last.day.Glicko','Animal','sexFeature']

        #concatenate the features
        X_scaled_df = pd.DataFrame(X_scaled_df, columns=feature_names, index=metadata.index)
        biomarkers_values = pd.concat([metadata,  X_scaled_df], axis=1)
        biomarkers_values_f = biomarkers_values[biomarkers_values["sex"] == "female"]
        biomarkers_values_m = biomarkers_values[biomarkers_values["sex"] == "male"] 

        for sheet_name, df in sheets.items():
            # Separate sexes
            df_female = df[df["sex"] == "female"]
            df_male = df[df["sex"] == "male"]

            feature_cols = [c for c in df.columns if c not in meta_cols]

            shap_values_f = df_female[feature_cols].values
            shap_values_m = df_male[feature_cols].values

            features_f = biomarkers_values_f[feature_cols]
            features_m = biomarkers_values_m[feature_cols]

                # --------- SORT FEATURES USING MALE SHAP IMPORTANCE ----------
            male_importance = np.mean(np.abs(shap_values_m), axis=0)
            sort_idx = np.argsort(male_importance)[::-1]

            feature_cols_sorted = np.array(feature_cols)[sort_idx]

            shap_values_m = shap_values_m[:, sort_idx]
            shap_values_f = shap_values_f[:, sort_idx]

            features_m = features_m[feature_cols_sorted]
            features_f = features_f[feature_cols_sorted]

            #add statistics to see if there are difference between males and females
            results_df =self.staticts_between_sex(shap_values_f, shap_values_m, feature_cols_sorted)
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            padj_dict = dict(zip(results_df["feature"], results_df["p_adj"]))

            # Create figure
            fig, (ax_male, ax_female) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

            # for females
            plt.sca(ax_female)
            shap.summary_plot(shap_values_f, features=features_f, feature_names=feature_cols_sorted, plot_type="violin", max_display=len(feature_cols_sorted), sort=False, show=False)
            ax_female.set_title("Female", fontsize=8)
            if self.run_model == 'linear':
                ax_female.set_xlim(-2.5,2.5)
                ax_female.set_xticks([-2.5,0,2.5])
            else:
                ax_female.set_xlim(-0.2,0.2)
                ax_female.set_xticks([-0.2,0,0.2])

            ax_female.set_xlabel(ax_female.get_xlabel(), fontsize=8)
            

            for tick in ax_female.get_yticklabels():
                tick.set_fontsize(6)

            for tick in ax_female.get_xticklabels():
                tick.set_fontsize(6)

            ax_female.axvline(0, color='gray', linewidth=0.5)

            #males
            plt.sca(ax_male)

            shap.summary_plot(shap_values_m, features=features_m, feature_names=feature_cols_sorted , plot_type="violin", max_display=len(feature_cols_sorted), sort=False, color_bar=False, show=False)

            ax_male.set_title("Male", fontsize=8)
            if self.run_model == 'linear':
                ax_male.set_xlim(-2.5,2.5)
                ax_male.set_xticks([-2.5,0,2.5])
            else:
                ax_male.set_xlim(-0.2,0.2)
                ax_male.set_xticks([-0.2,0,0.2])

            ax_male.set_xlabel(ax_male.get_xlabel(), fontsize=8)

            for tick in ax_male.get_yticklabels():
                tick.set_fontsize(6)

            for tick in ax_male.get_xticklabels():
                tick.set_fontsize(6)

            for tick in ax_female.get_yticklabels():
                 tick.set_fontsize(10)

            for tick in ax_male.get_yticklabels():
                tick.set_fontsize(10)

            ax_male.axvline(0, color='gray', linewidth=0.5)

            ####################add significance #################
            yticks = ax_male.get_yticks()
            yticklabels = [t.get_text() for t in ax_male.get_yticklabels()]
            for y, feature in zip(yticks, yticklabels):
                symbol = treat_continous_labels.significance_symbol(padj_dict[feature])
                if self.run_model == 'linear':
                    ax_male.text(2.5, y, symbol, ha="right", va="center", fontsize=14, fontweight="bold")
                else:
                    ax_male.text(0.2, y, symbol, ha="right", va="center", fontsize=14, fontweight="bold") 
               # ax_female.text(0, i, symbol, ha="left", va="center", fontsize=12, fontweight="bold")




            plt.suptitle(f"{sheet_name}", fontsize=10)
            plt.gcf().set_size_inches(20,10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_violin_' + sheet_name + '_plot.pdf'), dpi = 300, bbox_inches='tight')
            plt.close()
        writer.close()    
##############################statistics################
    def staticts_between_sex(self,shap_values_f, shap_values_m, feature_cols_sorted):
        results = []
        for i, feature in enumerate(feature_cols_sorted):
            stat, p = mannwhitneyu(shap_values_f[:,i], shap_values_m[:,i])
            results.append({ "feature": feature, "male_mean_abs_shap": np.mean(np.abs(shap_values_m[:,i])), "female_mean_abs_shap": np.mean(np.abs(shap_values_f[:,i])), "stat": stat, "p_value": p })

        results_df = pd.DataFrame(results)
        results_df["p_adj"] = multipletests(results_df["p_value"], method="fdr_bh")[1]  
        results_df["difference"] = results_df["male_mean_abs_shap"] - results_df["female_mean_abs_shap"] #say to which sex the prediction with the feature is push
        #save into excel 
       
        return results_df
    
    #####################save predictions #####################
    def save_predictions(self, y_true, y_pred):
        cols_true = ["Arch1_true", "Arch2_true", "Arch3_true", "Arch4_true"]
        cols_pred = ["Arch1_pred", "Arch2_pred", "Arch3_pred", "Arch4_pred"]
        cols_score  = ["Arch1_score", "Arch2_score", "Arch3_score", "Arch4_score"]

        df_true = pd.DataFrame(y_true, columns=cols_true)
        df_pred = pd.DataFrame(y_pred, columns=cols_pred)

        score = 1 - np.abs(y_true - y_pred)
        df_score = pd.DataFrame(score, columns=cols_score)

        df = pd.concat([df_true, df_pred, df_score], axis=1)
        filepath = os.path.join(self.output_dir, "ytrue_ypred.xlsx")
        df.to_excel(filepath, index=False)

    #### plot prediction score###############
    def plot_prediction_score(self, y_true, y_pred):
        archetypes = ["Arch1","Arch2","Arch3","Arch4"]

        score = 1 - np.abs(y_true - y_pred)

        mean_score = np.mean(score, axis=0)
        sem_score = np.std(score, axis=0) / np.sqrt(score.shape[0])
        
        plt.figure(figsize=(6,4))

        plt.bar(archetypes, mean_score, yerr=sem_score, capsize=5)

        plt.ylim(0,1)
        plt.ylabel("1 - |residuals|")
        plt.title("Prediction Accuracy per Archetype (mean ± SD)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "prediction_score_barplot.pdf"))
        plt.close()

    ### create confusion matrix of dominant archetype##############
    def confusion_matrix(self, y_true, y_pred):
            archetypes = ["Arch1","Arch2","Arch3","Arch4"]
            # Get dominant archetype (index of max)
            y_true_dom = np.argmax(y_true, axis=1)
            y_pred_dom = np.argmax(y_pred, axis=1)

            # Compute confusion matrix
            cm = confusion_matrix(y_true_dom, y_pred_dom)

            # Plot
            fig, ax = plt.subplots(figsize=(6,6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=archetypes)
            disp.plot(ax=ax, cmap="Blues", values_format="d")

            plt.title("Confusion Matrix (Dominant Archetype)")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confusion_matrix_dominant.pdf"))
            plt.close()

            # Optional: print accuracy
            accuracy = np.mean(y_true_dom == y_pred_dom)
            print(f"Dominant archetype accuracy: {accuracy:.3f}")

            return cm, accuracy
 ################# define significance######################3
    @staticmethod
    def significance_symbol(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "#"
        else:
            return ""

 ############# calculate metrics to approve model of continous variable############## 
    '''
     cosine similarity
  '''
    def evaluate_probabilistic_predictions(self, y_true, y_pred, plot = True):
        eps = 1e-10
        y_true_safe = np.clip(y_true, eps, 1) #try to avoid 0 prediction which can give problems with log operations
        y_pred_safe = np.clip(y_pred, eps, 1)
         # ----- Compute metrics -----
        cosine_sim = [1 - cosine(t, p) for t, p in zip(y_true, y_pred)]
        kl_div = np.sum(rel_entr(y_true_safe, y_pred_safe), axis=1)
        ent = entropy(y_pred.T)
        
        if plot:
            # ----- Create figure -----
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # ---- Cosine similarity ----
            axes[0].hist(cosine_sim, bins=20)
            axes[0].set_title(f"Cosine Similarity\nMedian = {np.median(cosine_sim):.3f}")
            axes[0].set_xlabel("Similarity")
            axes[0].set_ylabel("Count")

            # ---- KL divergence ----
            axes[1].hist(kl_div, bins=20)
            axes[1].set_title(f"KL Divergence\nMedian = {np.median(kl_div):.3f}")
            axes[1].set_xlabel("KL divergence")
            
            # ---- Entropy ----
            axes[2].hist(ent, bins=20)
            axes[2].set_title(f"Prediction Entropy\nMedian = {np.median(ent):.3f}")
            axes[2].set_xlabel("Entropy")


            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "probabilistic_evaluation.pdf"))
            plt.close()

        return np.median(cosine_sim), np.median(kl_div), np.median(ent)  



