import pandas as pd

class BehaviorDataLoader:
     def __init__(self, filepath):
        self.filepath = filepath

     def load_data(self):
         return pd.read_excel(self.filepath)