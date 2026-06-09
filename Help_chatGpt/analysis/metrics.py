from sklearn.metrics import (classification_report, confusion_matrix)
import pandas as pd

class  MetricsAnalyzer:
    def __init__(self, y_true, y_pred, class_names):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names

    def classification_report(self):
        target_names=[str(c) for c in self.class_names]
        report_dict =classification_report(self.y_true, self.y_pred, target_names=target_names,   output_dict=True)
        print(report_dict)
        report_df = pd.DataFrame(report_dict).transpose()
        print(report_df)
        return report_df


    def confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)

        cm_df = pd.DataFrame(

        cm,

        index=[
            f"True_{c}"
            for c in self.class_names
        ],

        columns=[
            f"Pred_{c}"
            for c in self.class_names
        ]
    )

        return cm, cm_df
