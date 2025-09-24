import pandas as pd


class rearrange_data: 
   def __init__(self, filename):
    self.filename = filename
    self.data = pd.DataFrame()
    self.data_selected = pd.DataFrame()

    
   def read_excel(self):
       df = pd.read_excel(self.filename)
       self.data = df
   
   def aggregate_data(self):
       self.data_selected = self.data.groupby(["Experiment", "sex", "Type", "Genotype", "Mice.chips", "Last.day.Glicko", "Animal", "Hierarchy" ]).mean().reset_index()
       
   def select_categories(self):
       #select alpha, beta, epsilon
       total_data =self.data_selected[(self.data_selected["Hierarchy"] == "alpha") |  (self.data_selected["Hierarchy"] == "beta") |  (self.data_selected["Hierarchy"] == "epsilon")].reset_index()
       return total_data
   
   def __call__(self):
       self.read_excel()
       self.aggregate_data()
       total_data_selection = self.select_categories()
       total_data = self.data_selected
       return total_data, total_data_selection
       
    