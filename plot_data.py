import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
from collections import defaultdict
import shap
import matplotlib as mpl
import matplotlib.collections as mcoll
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('TkAgg') 
import plot_violin_functions
#plt.ion() # Turn on the interactive mode


class plot_data:
    def __init__(self,data,title,output_directory,sex):
        self.data = data
        self.title = title
        self.output_directory = output_directory
        self.sex = sex
   
           
    def __call__(self,number_status, model_name):
        if number_status == 3: #concentrate in the alpha profile
          a=1
        elif number_status == 2:
          self.GetGraphs(model_name)
          
          
    '''
    input:
    output: all the graphs
    '''
    def GetGraphs(self, model_name):
      fig , axs = plt.subplots(2, 3, figsize=(10, 10))
      fig.suptitle(tuple(self.data[model_name]['classes']), fontsize=10)
      for i in range(6):
            axs[i // 3, i % 3].set_axis_off()
      #get graph of important features
      ax = axs[0,0]
      self.plotImportantFeatures(ax, model_name)
      #get dot graph
      ax = axs[0,1]
      self.plotDotFeatures(ax,model_name)
      #get violin 
      ax = axs[0,2]
      self.plotViolinFeatures(ax,model_name)
      #get important features which interact with the sex
      ax = axs[1,0]
      self.plotImportantFeaturesInteraction(ax,model_name)
      
      plt.tight_layout()
      plt.show()
      
    '''
    Important features
    '''
    
    def plotImportantFeatures(self,ax,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']

      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(),plot_type="bar",show=False)
      ax.set_title('Important features',fontsize=8)
      ax.set_xlabel("mean(|SHAP value|) \n (average impact on model output)")
      ax.xaxis.label.set_size(6)
      ax.yaxis.label.set_size(6)
      ax.set_xlim(0,1.5)
      subdivisions = [0,0.5,1,1.5]  # every 0.5 units
      ax.set_xticks(subdivisions)
      for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
      for tick in ax.get_xticklabels():
             tick.set_fontsize(6)
    
    '''
    Dot features
    '''  
    def plotDotFeatures(self,ax,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']
      
      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(),show=False)
      ax.set_title('Feature influence on prediction',fontsize=8)
      ax.set_xlabel("SHAP value \n (impact on model output)")
      ax.xaxis.label.set_size(6)
      ax.yaxis.label.set_size(6)
      for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
      for tick in ax.get_xticklabels():
             tick.set_fontsize(6)
      ax.set_xlim(-2.5,2.5)
      subdivisions = [-2.5,0,2.5]
      ax.set_xticks(subdivisions)
      
    '''
      Violin Features
      '''
    def plotViolinFeatures(self,ax,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']
      
      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(),plot_type = 'violin', show=False)
      ax.set_title('Feature influence on prediction',fontsize=8)
      ax.set_xlabel("SHAP value \n (impact on model output)")
      ax.xaxis.label.set_size(6)
      ax.yaxis.label.set_size(6)
      for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
      for tick in ax.get_xticklabels():
             tick.set_fontsize(6)
      ax.set_xlim(-2.5,2.5)
      subdivisions = [-2.5,0,2.5]
      ax.set_xticks(subdivisions)
      # Remove all PathCollection (scatter plots) from axes
      for collection in list(ax.collections):
             if isinstance(collection, mcoll.PathCollection):
                 collection.remove()
                 
    '''
     Features which interact with the sex
    '''
    def plotImportantFeaturesInteraction(self, ax,model_name):
       X = self.data[model_name]['data_features']
       features_names = X.columns
       sex_index = features_names.get_loc('sex_num')
       interaction_values1 = self.data[model_name]['interaction_shap']
       #since a list for each sample
       interaction_values = np.concatenate(interaction_values1, axis=0)
       sex_interactions = interaction_values[:, sex_index, :]
       #Remove self interactions
       sex_interactions[:, sex_index] = 0
       
       #graph
       plt.sca(ax)
       ax.set_axis_on()
       shap.summary_plot(sex_interactions, X , feature_names=X.columns.tolist(),plot_type="bar",show=False)
       ax.set_title('Important features',fontsize=8)
       ax.set_xlabel("mean(|SHAP value|) \n (average impact on model output)")
       ax.xaxis.label.set_size(6)
       ax.yaxis.label.set_size(6)
       ax.set_xlim(0,1.5)
       subdivisions = [0,0.5,1,1.5]  # every 0.5 units
       ax.set_xticks(subdivisions)
       for tick in ax.get_yticklabels():
              tick.set_fontsize(6)
       for tick in ax.get_xticklabels():
              tick.set_fontsize(6)
    