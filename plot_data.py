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
from Shap_functions import custom_summary_plot
from _violin import violin
from _labels import labels
from _violin_order import violin_order

matplotlib.use('TkAgg') 

#plt.ion() # Turn on the interactive mode


class plot_data:
    def __init__(self,data,title,output_directory,sex):
        self.data = data
        self.title = title
        self.output_directory = output_directory
        self.sex = sex
   
           
    def __call__(self,number_status, model_name,type_graph):
        if number_status == 3: #concentrate in the alpha profile
          a=1
        elif number_status == 2:
          self.GetFeaturesOrder(model_name)
          self.GetShapePerSex(model_name)
          self.GetGraphs(model_name,type_graph)
          
    '''
    input:
    output: all the graphs
    '''
    def GetGraphs(self, model_name,type_graph):
      match type_graph:
             case "shap_features_violin":
                   self.PlotShape(model_name) 
             case "random_permutation_plus_CM" :
                    self.Plot_random_cm(model_name) 
             case "shap_features_dotpoints":
                    self.shap_features_dotpoints(model_name)
          
    '''
    input: shap data plus features
    output: order that should be the features from most important- like self- according to all
    ''' 
    def GetFeaturesOrder(self,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']
      feature_names = X.columns.tolist()   
      
      #reorder features according to absolute value
      mean_abs_shap = np.abs(all_shap_values_1).mean(axis=0)
      self.top_indices = np.argsort(mean_abs_shap)[::-1]
      self.X_top = X.iloc[:,self.top_indices]
      self.shap_values_top = all_shap_values_1[:,self.top_indices]
      
      '''
      input: shap data
      output: shap for each sex
      '''
    def GetShapePerSex(self,model_name):
          all_shap_values = self.data[model_name]['shap_values']
          # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
          all_shap_values_1 = np.concatenate(all_shap_values, axis=0)   
          mice_information = pd.concat(self.data[model_name]['mice_information'], axis = 0)
          df_reset = mice_information.reset_index(drop = True) #reset the index
          # Find the row number(s)
          self.male_row_num = df_reset.index[df_reset['sex'] == 'male'].tolist()
          self.female_row_num = df_reset.index[df_reset['sex'] == 'female'].tolist()
          
          self.shap_values_males =  all_shap_values_1[self.male_row_num,:]
          self.shap_values_females =  all_shap_values_1[self.female_row_num,:]
          self.X_top_males = self.X_top.iloc[self.male_row_num]
          self.X_top_females = self.X_top.iloc[self.female_row_num,:]
          
    
    
       
   
    
    '''
    plot shape values
    '''
    def PlotShape(self, model_name):
      fig , axs = plt.subplots(2, 2, figsize=(20, 10))
      fig.suptitle(tuple(self.data[model_name]['classes']), fontsize=10)
      for i in range(4):
            axs[i // 2, i % 2].set_axis_off()
      #get graph of important features
      ax = axs[0,0]
      self.plotImportantFeatures(ax, model_name)
      #get violin all
      ax = axs[0,1]
      self.plotViolinFeaturesCustom(ax,model_name, 'all')            
      #get violin male
      ax = axs[1,0]     
      self.plotViolinFeaturesCustom_select(ax,model_name, 'male')
      #get violin female
      ax = axs[1,1]     
      self.plotViolinFeaturesCustom_select(ax,model_name, 'female')
      
      plt.tight_layout()
      #plt.show()
      plt.savefig(self.output_directory + self.title  +'.pdf', format='pdf',dpi=300,bbox_inches='tight') 
    
    '''
    Plot metrics data
    '''
    def Plot_random_cm(self, model_name): 
           
           fig , axs = plt.subplots(2, 2, figsize=(10, 10))
           fig.suptitle(tuple(self.data[model_name]['classes']), fontsize=10)  
           for i in range(4):
            axs[i // 2, i % 2].set_axis_off()
           #get random permutation
            ax = axs[0,0]
            self.PlotRandomPermutation(ax, model_name,0) #class 0
            ax = axs[0,1]
            self.PlotRandomPermutation(ax, model_name,1) #class 1
            ax = axs[1,0]
            self.PlotConfusionMatrix(ax,model_name)
            
            plt.tight_layout()
            #plt.show()
            plt.savefig(self.output_directory + self.title  +'_random_permutation_CM.pdf', format='pdf',dpi=300,bbox_inches='tight') 
     
    '''
     Plot features and dot points
     '''       
    def shap_features_dotpoints(self, model_name):
           fig , axs = plt.subplots(1, 2, figsize=(4, 4))
           fig.suptitle(tuple(self.data[model_name]['classes']), fontsize=10)  
           axs[0].set_axis_off()  
           axs[1].set_axis_off()
           #get important features
           ax = axs[0]
           self.plotImportantFeatures(ax, model_name)
           #get dot plots
           ax = axs[1]
           self.plotDotFeatures(ax,model_name)
            
           plt.tight_layout()
            #plt.show()
           plt.savefig(self.output_directory + self.title  +'_Features_dotPoints.pdf', format='pdf',dpi=300,bbox_inches='tight') 
           
            
            
    '''
    Important features
    '''
    
    def plotImportantFeatures(self,ax,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']
      feature_names = X.columns.tolist()
      
      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(),plot_type="bar",max_display=len(feature_names), show=False)
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
    Custom violin plot
    '''
    
    def plotViolinFeaturesCustom(self,ax,model_name,sex):
           
      if sex == 'all':
         features_names = self.X_top.columns.tolist()
         shap_values =   self.shap_values_top  
         features = self.X_top    
      elif  sex == 'male':
             shap_values =   self.shap_values_males[:,self.top_indices]  #order according to the top indices found with all
             features_names = self.X_top.columns.tolist()
             features = self.X_top_males    
      elif sex == 'female': 
           shap_values =   self.shap_values_females[:,self.top_indices]  
           features_names = self.X_top.columns.tolist()
           features = self.X_top_females   
         
      #graph
      ax.set_axis_on()
      plt.sca(ax)
      
      
      violin(
    shap_values,
    features= features,
    feature_names=features_names,
    max_display=len(features_names),
    plot_type = "violin",
    sort=False, show=False
    )
      
      for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
      for tick in ax.get_xticklabels():
             tick.set_fontsize(6)
      ax.set_title(sex,fontsize=8)
      ax.set_xlim(-2.5,2.5)
      subdivisions = [-2.5,0,2.5]
      ax.set_xticks(subdivisions)
    
    '''
    Custom violin plot for females
    '''
    
    def plotViolinFeaturesCustom_select(self,ax,model_name,sex):
           
     
       features_names = self.X_top.columns.tolist()
       shap_values =   self.shap_values_top  
       features = self.X_top  
         
       if sex == 'male':
             select_index = self.male_row_num  
       elif sex == 'female':   
           select_index = self.female_row_num
      #graph
       ax.set_axis_on()
       plt.sca(ax)
      
      
       violin_order(
    shap_values,
    features= features,
    feature_names=features_names,
    max_display=len(features_names),
    plot_type = "violin",
    sort=False, show=False, select_index = select_index,
    )
      
       for tick in ax.get_yticklabels():
              tick.set_fontsize(6)
       for tick in ax.get_xticklabels():
              tick.set_fontsize(6)
       ax.set_title(sex,fontsize=8)
       ax.set_xlim(-2.5,2.5)
       subdivisions = [-2.5,0,2.5]
       ax.set_xticks(subdivisions) 
    
    '''
    Custom violin plot for males
    '''
    
    def plotViolinFeaturesCustom_males(self,ax,model_name,sex):
           
      if sex == 'all':
         features_names = self.X_top.columns.tolist()
         shap_values =   self.shap_values_top  
         features = self.X_top    
      elif  sex == 'male':
             shap_values =   self.shap_values_males[:,self.top_indices]  #order according to the top indices found with all
             features_names = self.X_top.columns.tolist()
             features = self.X_top_males    
      elif sex == 'female': 
           shap_values =   self.shap_values_females[:,self.top_indices]  
           features_names = self.X_top.columns.tolist()
           features = self.X_top_females   
         
      #graph
      ax.set_axis_on()
      plt.sca(ax)
      
      
      violin(
    shap_values,
    features= features,
    feature_names=features_names,
    max_display=len(features_names),
    plot_type = "violin",
    sort=False, show=False
    )
      
      for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
      for tick in ax.get_xticklabels():
             tick.set_fontsize(6)
      ax.set_title(sex,fontsize=8)
      ax.set_xlim(-2.5,2.5)
      subdivisions = [-2.5,0,2.5]
      ax.set_xticks(subdivisions) 
    
    
    
    
    
    ######################################No used
    
    '''
    Dot features
    '''  
    def plotDotFeatures(self,ax,model_name):
      all_shap_values = self.data[model_name]['shap_values']
      # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
      all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
      X = self.data[model_name]['data_features']
      feature_names = X.columns.tolist()
      
      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(),max_display=len(feature_names),show=False)
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
      feature_names = X.columns.tolist()
      
      #graph
      plt.sca(ax)
      ax.set_axis_on()
      shap.summary_plot(all_shap_values_1, X , feature_names=X.columns.tolist(), max_display=len(feature_names),show=False,sort = False)
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
    
    
    '''
    plot random permutation
    '''
    def PlotRandomPermutation(self,ax, model_name,index):
           
           class_ = self.data[model_name]['classes']
           plt.sca(ax) 
           ax.set_axis_on()
           fscore_actual = (self.data[model_name]['fscore'])[index]
           class_num =str(index +1)
           pval = np.mean(self.data[model_name]['shuffle_fscore_class'+ class_num] >= fscore_actual)
           plt.hist(self.data[model_name]['shuffle_fscore_class'+ class_num ], bins=6, alpha=0.7, color='blue', edgecolor='black')#bin 6
           plt.axvline(fscore_actual, color='red',linestyle='--', label= f"Actual F-score = {fscore_actual}")
           plt.text(fscore_actual + 0.1, plt.ylim()[1]*0.8, f"pvalue = {pval:.3f}", color='red',fontsize = 6)
           ax.set_title('random permutation',fontsize=8)
           ax.set_xlim(0,1)
           ax.set_ylabel("Frequency")
           ax.set_xlabel('shuffle_fscore_' + class_[index])
           ax.xaxis.label.set_size(6)
           ax.yaxis.label.set_size(6)
           for tick in ax.get_yticklabels():
             tick.set_fontsize(6)
           for tick in ax.get_xticklabels():
             tick.set_fontsize(6)  
             
    '''
   Plot confusion matrix
   '''        
    def PlotConfusionMatrix(self,ax,model_name):
        try:
                  cm = self.data[model_name]['confusion_matrix']
                  #get classes
                  class_names = self.data[model_name]['classes']
                  #divide each number by its row sum
                  cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #new axis row row
             
                  ax.set_axis_on()
                  #Normalize by the total number of instances per class
                  sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 8},ax =ax)
                  
                  ax.set_xlabel('Predicted Labels')
                  ax.set_ylabel('True Labels')
        except Exception as e:
                  print("error")   