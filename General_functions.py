import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.preprocessing import StandardScaler

class General_functions:
   def __init__(self,input_file):
         self.input_file = input_file
         self.data = pd.read_excel(input_file,sheet_name="All_data")
   
   '''
    input: data sex, choice, hormones panel architype
    select: the data either for hierarchy or architype
    '''
   def select_data(self,sex = None, hormones = None):
        #select sex
        
        selected_data = pd.DataFrame()
        if sex == "all":
          selected_data = self.data
        else:
          selected_data = self.data[self.data['sex'] == sex]
       
        selected_data['status'] = selected_data['Hierarchy']
         
        #select hormones and status
       
            #add a columns to distinguish each arena experiment with numbers, beggining from 1
        selected_data = selected_data.copy()
        selected_data['groups'] =  selected_data['Experiment'].str[-3:]
        selection = hormones[:] #creates a shallow copy
        selection.append('status')
        selection.append('groups')
        selected_data = selected_data.loc[:,selection]
            
            
     
        return selected_data
    
   '''
    input : data
    output: add ratios if there are
    '''
   @staticmethod
   def addRatios(data, hormones):
        result = [item for item in hormones if '_' in item]
        if not result:
          print("The list is empty")
        else:
         for r in result:
          parts = r.split('_')
          first_part = parts[0]
          last_part = parts[-1]
          data[r] = data[first_part]/data[last_part]
        return data  
    
   '''
    input: data
    output: tsne graph
    '''
   @staticmethod
   def tsne_plot(data,sex):
       #create a t-SNE object
       tsne = TSNE(n_components = 2, perplexity = 5, n_iter=2000, random_state=42)
       #select all the features
       features = data.iloc[:,:-2]
       # Normalize the data using StandardScaler
    #    scaler = StandardScaler()
    #    X_normalized = scaler.fit_transform(features)

       #fit t-sne on the data and transform it
       X_tsne = tsne.fit_transform(features)
       #labels is the status
       labels = data.iloc[:,-2]
       text_labels = data.iloc[:,-1]
       color_map ={'alpha': 'red', 'beta':'blue','gamma':'green','delta':'yellow','epsilon': 'cyan'}
       labels_c =[color_map[i] for i in labels]
       # Plot the 2D representation with colors based on the labels
       plt.figure(figsize=(8, 6))
       plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_c)
       
       legend_handles = [mpatches.Patch(color=color_map[label], label=f' {label}')
                  for label in np.unique(labels)]
       
       
# Add labels to each point
       for i, txt in enumerate(text_labels.tolist()):
               plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=9, ha="right", va="bottom")
              
       plt.legend(handles=legend_handles, title=" ")
       plt.title("t-SNE Visualization with hierarchy data "+ sex)
       plt.xlabel("Dimension 1")
       plt.ylabel("Dimension 2")
       
       plt.show()
       
       
    
   def  __call__(self, sex = None, hormones = None):  
       #add ratios if there are
       self.data = General_functions.addRatios(self.data,hormones) 
       selected_data = self.select_data(sex, hormones) 
       General_functions.tsne_plot(selected_data,sex)
       a=1