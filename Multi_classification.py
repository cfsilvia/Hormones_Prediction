
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from learning_data import learning_data
from sklearn.preprocessing import RobustScaler
import pickle





class MultiClassification:

    def __init__(self, data, output_dir, type_model = None, number_archetypes = 3, normalization = True, select_pairs_mode = None):
        self.data = data
        self.output_dir = output_dir
        self.model_names = type_model
        self.number_archetypes = number_archetypes
        self.par_normalization = normalization
        self.learning_results = {}
        self.select_pairs_mode = select_pairs_mode



    def __call__(self):
        
        #load data and  select the data
        datasets, metadata = self.load_data()

        for model_name in self.model_names:
            print(f"Running model: {model_name}")
        
            self.learning_results[model_name] = {}
            for index in range(1, self.number_archetypes + 1):
                self.learning_results[model_name][f'arch{index}'] = self.train_learning(datasets[f'arch{index}'], archetype_name=f'arch{index}', information_data=metadata,
                                                                                        model_name=model_name)
                
                # run shuffle test
                shuffle_f1_class0, shuffle_f1_class1 = self.add_shuffling(datasets[f'arch{index}'],archetype_name=f'arch{index}',
                                     model_name=model_name, n_iterations=200)
                
                # save shuffle results
                self.learning_results[model_name][f'arch{index}']["shuffle_fscore_rest"] = shuffle_f1_class0
                self.learning_results[model_name][f'arch{index}']["shuffle_fscore_archetype"] = shuffle_f1_class1



        #save as pkl
        filename = self.output_dir + 'learning_results.pkl'
        with open(filename, "wb") as f:
            pickle.dump(self.learning_results, f)
            f.close()
        
        filename_for_excel = self.output_dir + 'metrics_results.xlsx'
        self.save_summary_to_excel(filename_for_excel)
            



        
     ################ LOAD DATA ################

    def load_data(self):

        data = pd.read_excel(self.data)
        
        metadata_cols = ['Experiment','sex','Type','Genotype','Hierarchy','Mice.chips','Last.day.Glicko','Animal']

        metadata = data[metadata_cols]

        # Generate archetype column names dynamically
        archetype_cols = [f'Arch{i+1}' for i in range(self.number_archetypes)]

        # Drop metadata + archetype + optional extra columns safely
        drop_cols = metadata_cols + archetype_cols + ['sexFeature']
        drop_cols = [col for col in drop_cols if col in data.columns]  # avoid errors

        X = data.drop(columns=drop_cols, axis=1)
        
       # Select archetype targets dynamically
        y = data[archetype_cols].values

        if self.select_pairs_mode == "near_far":
            print("Selecting data using near-far method...")
            datasets = self.select_data_near_far(X, y, metadata)
        else:
            datasets = self.select_data(X, y, self.number_archetypes)

        self.save_datasets_to_excel(datasets, metadata, self.output_dir + "archetypes.xlsx")

        feature_names = X.columns.tolist()

        return datasets , metadata
    
    ################# SELECT DATA NEAR-FAR ################
    def select_data_near_far(self, X, y, metadata):
        datasets = {}
        n = 11  # number of samples to select for near and far

        for a in range(self.number_archetypes):
            p = y[:, a]
            sorted_idx = np.argsort(p)
             # Lowest n
            low_idx = sorted_idx[:n]
            # Highest n
            high_idx = sorted_idx[-n:][::-1]  # reverse for descending

            # Select features
            X_low = X.iloc[low_idx]
            X_high = X.iloc[high_idx]

            # Labels
            y_low = np.zeros(len(low_idx))   # far / low
            y_high = np.ones(len(high_idx)) # near / high

             # Combine
            X_pair = pd.concat([X_high, X_low])
            y_pair = np.concatenate([y_high, y_low])

            # Optional: use p values as weights
            weights = np.concatenate([p[high_idx], p[low_idx]])

            # ✅ metadata per archetype
            metadata_subset = metadata.iloc[np.concatenate([high_idx, low_idx])]

            datasets[f"arch{a+1}"] = (X_pair, y_pair, weights, metadata_subset)

        return datasets



    
    ################ SELECT DATA ################
    def select_data(self, X, y, number_archetypes):
         datasets = {}
         for a in range(number_archetypes):
             p = y[:, a]
             others = np.delete(y, a, axis=1)
             max_other = others.max(axis=1)
             margin = p - max_other
            
             if number_archetypes == 3:
               pnear = 0.4
               pfar = (1/6)

             near_mask = (margin > 0) & (p > pnear)
             far_mask = (margin < 0) & (p <= pfar)
             
             # extract data
             Xnear_features = X[near_mask]
             pnear_vals = p[near_mask]

             Xfar_features = X[far_mask]
             pfar_vals = p[far_mask]

             # balance
             n = min(len(Xnear_features), len(Xfar_features))
             n=11
            
             # far: smallest p first
             far_sorted_idx = np.argsort(pfar_vals)
             selected_far_idx = far_sorted_idx[:n]

             # near: largest p first
             near_sorted_idx = np.argsort(-pnear_vals)
             selected_near_idx = near_sorted_idx[:n]

             Xnear_features = Xnear_features.iloc[selected_near_idx]
             Xfar_features = Xfar_features.iloc[selected_far_idx]
            

             y_near = np.ones(n)      # label 1 for near
             y_far  = np.zeros(n)     # label 0 for far

             X_pair = pd.concat([Xnear_features, Xfar_features])
             y_pair = np.concatenate([y_near, y_far])
             weights = np.concatenate([pnear_vals[selected_near_idx], pfar_vals[selected_far_idx]])

             datasets[f"arch{a}"] = (X_pair, y_pair,weights)

         return datasets
           
    ################## SAVE DATASETS TO EXCEL ################
    def save_datasets_to_excel(self, datasets, metadata, filename):
        with pd.ExcelWriter(filename) as writer:
            for key, data in datasets.items():
                X = data[0]
                y = data[1]
                w = data[2]

                # select matching metadata rows
                metadata_subset = metadata.loc[X.index]

                ## combine everything
                df = pd.concat([metadata_subset, X.copy()], axis=1)
                df["label"] = y
                df["weight"] = w
                # write to a separate sheet
                df.to_excel(writer, sheet_name=key, index=False)

    



    ################ MODEL ################

    def build_model(self):
        #     model = RandomForestClassifier(
        #     n_estimators=300,
        #     max_depth=None,
        #     min_samples_leaf=2,
        #     class_weight='balanced',
        #     random_state=42
        # )

            model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    class_weight='balanced'
     )
            return model
    
   #######################Learning########################
    '''
     input: selected data
    output : after splitting and learning get dictionary 
    '''
    def train_learning(self,dataset,archetype_name, information_data = None, model_name = None):
         
         X, y, weights, information_data= dataset
         # Set up Leave-One-Out Cross-Validation (LOOCV)
         loo = LeaveOneOut()
         
        # Accumulators
         predictions = []
         predicted_probs = []
         true_labels = []
         all_shap_values = []
         all_interaction_values = []
         all_mice_information = []



         # LOOCV Loop: For each iteration, one sample is held out as the test sample.
         for train_idx, test_idx in loo.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if information_data is not None:
                information_mice = information_data.iloc[test_idx]
            # #normalize the train data-use normalization for test data
            if self.par_normalization:
                X_train_scaled, X_test_scaled = self.normalization(X_train,X_test)
            else:
                X_train_scaled = X_train.to_numpy()
                X_test_scaled = X_test.to_numpy()
                
            #balance the train data by using smote  with the larger class
           # X_train_resampled, y_train_resampled = self.balance_data(X_train_scaled,y_train)
           # X_train_resampled, y_train_resampled = (X_train_scaled,y_train)
            # #check before
            learner = learning_data(X_train_scaled, X_test_scaled,y_train, y_test,model_name,number_labels =2 )
                
            y_pred, y_prob, y_test, shap_values, interaction_values = learner()
                
            # Store results
            predictions.append(int(y_pred[0]))
            predicted_probs.append(y_prob[0].tolist())
            true_labels.append(int(y_test[0]))
            all_shap_values.append(shap_values)
            all_interaction_values.append(interaction_values)

            if information_data is not None:
                all_mice_information.append(information_mice)

  
    #      #get shap values of the final model with stable features
    #  #    shap_values = Find_better_features.GetShapValues(X[stable_feature_names],y,model_name)

         # Metrics
         accuracy, precision, recall, f1, cm, balanced_acc, fpr, tpr, thresholds, roc_auc = \
                             learning_data.metrics(predictions, predicted_probs, true_labels)
         # Results dictionary
         results_dict = {
            "classes": archetype_name,  # assumed external
            "features": X.columns,
            "prob": predicted_probs,
            "labels_pred": predictions,
            "true_labels": true_labels,
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "fscore": f1,
            "shap_values": all_shap_values,
            "interaction_shap": all_interaction_values,
            "data_features": X,
            "mice_information": all_mice_information,
            "FPR": fpr,
            "TPR": tpr,
            "thresholds": thresholds,
            "roc_auc_metrics": roc_auc,
        }

    
         return results_dict

###############normalization###########################
    '''
    input = train and test features data
    output = normalized train data , and test data normalize as train data
              each column is normalized independent of the others
    ''' 
    def normalization(self,X_train,X_test):
        scaler = RobustScaler()  #change to   robust  for the case of ouliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled



#############################################################
    ################ RUN MODEL ################
    def run_model(self, X, y, feature_names, metadata):

        all_y_true = []
        all_y_pred = []

        acc_scores = []
        bal_acc_scores = []
        logloss_scores = []
        cms = []

        # Define the cross-validation strategy
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

        # Loop over splits
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale BEFORE SMOTE for logistic regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

             # Apply SMOTE only on training data
            # k_neighbors must be < smallest class count in training fold
            smote = SMOTE(k_neighbors=3, random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)



            model = self.build_model()
            model.fit(X_train_res, y_train_res)

            # Predict
             # Predict on untouched test fold
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            ll = log_loss(y_test, y_proba, labels=[0, 1, 2])

            acc_scores.append(acc)
            bal_acc_scores.append(bal_acc)
            logloss_scores.append(ll)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            cms.append(cm)

            
            print(
                f"Fold {fold_idx+1}: "
                f"Accuracy={acc:.3f}, "
                f"BalancedAccuracy={bal_acc:.3f}, "
                f"LogLoss={ll:.3f}"
            )


        # =========================
        # Final results
        # =========================
        print("\n==== FINAL RESULTS ====")
        print(f"Mean Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
        print(f"Mean Balanced Accuracy: {np.mean(bal_acc_scores):.3f} ± {np.std(bal_acc_scores):.3f}")
        print(f"Mean LogLoss: {np.mean(logloss_scores):.3f} ± {np.std(logloss_scores):.3f}")

        # Aggregate confusion matrix across all folds
        total_cm = np.sum(cms, axis=0)
        print("\n==== CONFUSION MATRIX (summed over folds) ====")
        print(total_cm)

        cm_df = pd.DataFrame(
            total_cm,
            index=["True_0", "True_1", "True_2"],
            columns=["Pred_0", "Pred_1", "Pred_2"]
        )
        print("\n", cm_df)

        print("\n==== CLASSIFICATION REPORT ====")
        print(classification_report(all_y_true, all_y_pred, labels=[0, 1, 2]))

        # Plot summed confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=[0, 1, 2])
        disp.plot()
        plt.title("Confusion Matrix (summed over CV folds)")
        plt.show()

        return acc_scores, bal_acc_scores, logloss_scores
    
    # Run pairwise classification for each pair of classes
    def run_pairs(self, X, y, feature_names, metadata):
        pairs = [(0, 1), (0, 2), (1, 2)]
        
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        model = LogisticRegression(max_iter=1000)

        for a, b in pairs:
            mean_acc, std_acc = self.run_pairwise_model(X, y, a, b, model, cv)
            print(f"{a} vs {b}: {mean_acc:.3f} ± {std_acc:.3f}")
                
    #run pairwise classification for a specific pair of classes
    def run_pairwise_model(self,X, y, class_a, class_b, model, cv):
        mask = (y == class_a) | (y == class_b)
        
        X_pair = X[mask]
        y_pair = y[mask]
        
        # relabel to 0/1
        y_pair = (y_pair == class_a).astype(int)

        scores = []

        for train_idx, test_idx in cv.split(X_pair, y_pair):
            X_train, X_test = X_pair.iloc[train_idx], X_pair.iloc[test_idx]
            y_train, y_test = y_pair[train_idx], y_pair[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        return np.mean(scores), np.std(scores)   

    ############save results in excel#######################
    def save_summary_to_excel(self, filename):
        rows = []
        for model_name, model_data in self.learning_results.items():
           for arch_name, results in model_data.items():

            precision = results["precision"]
            recall = results["recall"]
            fscore = results["fscore"]
            cm = results["confusion_matrix"]

            row = {
                "model": model_name,
                "archetype": arch_name,
                "confusion_matrix": cm,
                "accuracy": round(results["accuracy"], 3)*100,
                "balanced_accuracy": round(results["balanced_accuracy"], 3)*100,
                "roc_auc": round(results["roc_auc_metrics"], 3),

                # class 0 = rest
                "precision_rest": round(precision[0], 3)*100,
                "recall_rest": round(recall[0], 3)*100,
                "fscore_rest": round(fscore[0], 3)*100,

                # class 1 = archetype
                "precision_archetype": round(precision[1], 3)*100,
                "recall_archetype":  round(recall[1], 3)*100,
                "fscore_archetype": round(fscore[1], 3)*100,
            }

            rows.append(row)

    
        df = pd.DataFrame(rows)
        
   

        df.to_excel(filename, index=False)

 #################### SHUFFLING TEST ####################
    '''
    input_data: shuffled labels
    output_data: F-score for each shuffle
    '''
    def add_shuffling(self, dataset, archetype_name=None, model_name=None, n_iterations=1000):            
         X, y, weights, information_data = dataset

         all_fscore_class1 = []
         all_fscore_class2 = []
         # fixed RNG for reproducibility
         rng = np.random.default_rng(42)

         for i in range(n_iterations):
            print(f"Shuffle iteration {i+1}/{n_iterations}")
            # shuffle labels only
            perm = rng.permutation(len(y))
            y_perm = y[perm]
            # rebuild dataset with shuffled labels
            shuffled_dataset = (X, y_perm, weights, information_data)

             # train model with shuffled labels
            results_dict = self.train_learning(
                shuffled_dataset,
                archetype_name=archetype_name,
                information_data=information_data,
                model_name=model_name)
            # store f-scores for class 1 and class 2
            fscore = results_dict['fscore']

            all_fscore_class1.append(fscore[0])
            all_fscore_class2.append(fscore[1])
         return all_fscore_class1, all_fscore_class2

