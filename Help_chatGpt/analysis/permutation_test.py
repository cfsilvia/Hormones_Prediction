import matplotlib.pyplot as plt
from sklearn.model_selection import (RepeatedStratifiedKFold, permutation_test_score)
from sklearn.metrics import (f1_score, make_scorer)
import numpy as np


class PermutationTest:
     def __init__(self, model, X, y):
          self.model = model
          self.X = X
          self.y = y
          self.outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
     
     def run_permutation_test(self):
          score, permutation_scores, pvalue = (permutation_test_score(estimator=self.model, X=self.X, y=self.y,
                                    cv=self.outer_cv, scoring="f1_macro", n_permutations=1000, random_state=42,n_jobs=-1 ))
          print("\n" + "=" * 60)

          print("PERMUTATION TEST")

          print("=" * 60) 

          print(f"\nObserved Macro F1: {score:.3f}")

          print(f"p-value: {pvalue:.5f}")

          return {

        "score": score,

        "pvalue": pvalue,

        "permutation_scores": permutation_scores
    }    
     
     def run_permutation_test_per_class(self, class_names):
          results = {}
          print("\n" + "=" * 60)
          print("CLASS-SPECIFIC PERMUTATION TESTS")
          print("=" * 60)

          for idx, class_name in enumerate(class_names):
               #Can the model distinguish this class from all the others?
               binary_y = (self.y == idx).astype(int) #create binary labels for the current class vs all others 1 vs 0
               score, permutation_scores, pvalue = (permutation_test_score(estimator=self.model, X=self.X, y=binary_y,
                                    cv=self.outer_cv, scoring=make_scorer(f1_score, average="binary"), n_permutations=1000, random_state=42,n_jobs=-1 ))
               print(f"\nClass: {class_name}")
               print(f"Observed F1: {score:.3f}")
               print(f"p-value: {pvalue:.5f}")
               results[class_name] = {
                    "score": score,
                    "pvalue": pvalue,
                    "permutation_scores": permutation_scores
               }
          return results