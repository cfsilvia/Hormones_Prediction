import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

class statistics_class:
    def __init__(self, input_file, initial, final):
        self.input_data = pd.read_excel(input_file, sheet_name="Sheet1")
        self.initial = initial-1
        self.final = final
        self.output_file = input_file

    def __call__(self):
      #separate females from males
      fem = self.input_data[self.input_data['sex'] == "female"].iloc[:,self.initial:self.final]
      mal = self.input_data[self.input_data['sex'] == "male"].iloc[:,self.initial:self.final]
      self.compare_sex(fem, mal)

    '''
    apply mann whitney
    '''
    def compare_sex(self,fem, mal):
     results = []
     molecules = fem.columns
     for col in range(len(fem)+len(mal)):
        U, p = mannwhitneyu(fem.iloc[:,col], mal.iloc[:,col], alternative="two-sided")
        results.append({"molecule": molecules[col], "p_value": p, "sign_pvalue": statistics_class.mark_sig(p), "median_female": np.median(fem),
                       "median_male": np.median(mal) })
     out = pd.DataFrame(results)
     with pd.ExcelWriter(self.output_file, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        out.to_excel(writer, sheet_name="MannWhitney_Pvalues", index=False)

    '''
    '''
    @staticmethod
    def mark_sig(p):
       if p < 0.001:
           return "***"
       elif p < 0.01:
            return "**"
       elif p < 0.05:
            return "*"
       elif p <= 0.10:
            return "#"
       else:
            return ""
        