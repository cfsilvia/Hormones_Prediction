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
#plt.ion() # Turn on the interactive mode


class plot_data:
    def __init__(self,data,title,output_directory,sex):
        self.data = data
        self.title = title
        self.output_directory = output_directory
        self.sex = sex
    '''
    input:
    output: prob histogram
    ''' 
    def plot_prob(self,  hormone_dict,select_column_prob):
      num_keys = len(hormone_dict)
      num_items = [len(v) for v in hormone_dict.values()]
      fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1), figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60 figsize=(20, 40)
      fig.subplots_adjust(hspace=1, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
      for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
       
      for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_prob = self.data[h][m]['prob'] 
                first_elements = [array[:,select_column_prob] for array in list_prob]#for first class
                list_labels = self.data[h][m]['labels_pred'] 
                class_names = self.data[h][m]['classes'] 
                #extract those  values for alpha and those for beta
                #take the probability of first class
                first_elements_list = plot_data.extract_first_class(first_elements)
                first_elements_labels= plot_data.extract_first_class(list_labels)
                # Convert counts to frequencies
                #frequencies = [(count / sum(first_elements_list))*100 for count in first_elements_list]
                first_elements_labels = [class_names[0] if label == 0 else class_names[1] for label in first_elements_labels]
                
                # Create a DataFrame
                df = pd.DataFrame({'Category': first_elements_labels, 'Value': first_elements_list } )
                sns.histplot(data = df ,x='Value', hue='Category', bins=100, kde=False,stat="count", palette={class_names[0]: 'red', class_names[1]: 'blue'}, multiple="stack",ax =axs[i,j])
                #sns.histplot(x=first_elements_list, bins=100, kde=True, color = 'green',ax =axs[i,j])
                # Customize the legend
                legend_elements = [
                    Line2D([0], [0], color='red', lw=2, label=class_names[0]),
                    Line2D([0], [0], color='blue', lw=2, label=class_names[1])
                ]
                axs[i,j].legend(title="",handles = legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
               # plt.title("Grouped Histogram")
                axs[i,j].set_xlabel("Prob." + str(class_names[select_column_prob]))
                axs[i,j].set_ylabel("counts")

      fig.tight_layout(pad=5)
      fig.suptitle('Histogram'+ '  ' + self.title, x=0.5, y=0.99) 
      plt.savefig(self.output_directory + self.title + '_Histogram_Probability' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
      #plt.show()
            
                

                
    @staticmethod   
    def extract_first_class(first_elements):
        flattened_array = np.concatenate(first_elements)
        first_elements_list = flattened_array.tolist() 
        return  first_elements_list 
          
                
    '''
    input:
    output: precision bar plot
    '''
    def plot_precision(self, hormones_list,models_list):
        num_keys = len(hormones_list)
        num_items = len(models_list)
        
        fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
        for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
       
        i=0
        for j, m in enumerate(models_list):
                try:
                  list_precision = self.data[m]['precision']   
                  array_mean =100*np.mean(list_precision, axis =0) 
                  array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                  array_mean_list = array_mean.tolist()
                  array_se_list = array_se.tolist()
                  #get classes
                  class_names = self.data[m]['classes']
                  axs[i,j].set_axis_on()
                  #bar plot
                  axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                  axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
                  axs[i, j].set_ylabel("Precision % ", fontsize = 8) 
                  axs[i,j].set_ylim(0, 120)
                except Exception as e:
                  print("continue")
        
                fig.tight_layout(pad=5)
                fig.suptitle('Precision: TP/(TP+FN)'+ '  ' + self.title, x=0.5, y=0.99) 
                plt.savefig(self.output_directory + self.title + '_precision' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       #plt.show()
            
    '''
    input:
    output: accuracy bar plot
    '''
    def plot_accuracy(self, hormones_list,models_list):
       num_keys = len(hormones_list)
       num_items = len(models_list)
       fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
       fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
       for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
               
           i=0
           for j, m in enumerate(models_list):
              try:
                list = self.data[m]['accuracy']   
                array_mean =100*np.mean(list) 
                #get classes
                class_names = self.data[m]['classes']
                axs[i,j].set_axis_on()
                #bar plot
                axs[i,j].bar("two_classes", array_mean.tolist(), color=['skyblue'], edgecolor = 'black',width = 0.01)
                axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Accuracy % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
              except Exception as e:
                  print("continue")
          
       fig.tight_layout(pad=5)
       fig.suptitle('Accuracy: '+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_accuracy' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       #plt.show()
            
    '''
    input:
    output: recagg bar plot
    '''
    def plot_recall(self, hormones_list,models_list):
      num_keys = len(hormones_list)
      num_items = len(models_list)
      fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
      fig.subplots_adjust(hspace=0.5, wspace=0.3)  #
        # turn off the axes
      for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
       
           i=0
           for j, m in enumerate(models_list):
              try:
                list_recall = self.data[m]['recall']   
                array_mean =100*np.mean(list_recall, axis =0) 
                array_se = 100*np.std(list_recall , axis=0, ddof=1)/np.sqrt(len(list_recall ))
                array_mean_list = array_mean.tolist()
                array_se_list = array_se.tolist()
                
                #get classes
                class_names = self.data[m]['classes']
                axs[i,j].set_axis_on()
                #bar plot
                axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Recall % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
              except Exception as e:
                  print("continue")
       
      fig.tight_layout(pad=5)
      fig.suptitle('Recall: TP/(TP+FP)'+ '  ' + self.title, x=0.5, y=0.99) 
      plt.savefig(self.output_directory + self.title + '_recall' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
      #plt.show()
    '''
    input:
    output: plot confusion matrix
    '''
    def plot_confusion_matrix(self, hormones_list,models_list):
        num_keys = len(hormones_list)
        num_items = len(models_list)
        
        fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
        for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
        
        #fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 15))
        
        
        
        for j, m in enumerate(models_list):
                i=0
                try:
                  list_confusion_matrix = self.data[m]['confusion_matrix']
                  #get classes
                  class_names = self.data[m]['classes']
                  # Calculate the sum of all the confusion matrix
                  sum_cm = np.sum(list_confusion_matrix, axis=0)
                  #calculate the sum of the rows
                  sum_rows = sum_cm.sum(axis=1, keepdims=True) 
                  axs[i,j].set_axis_on()
                  #Normalize by the total number of instances per class
                  sns.heatmap(sum_cm/sum_rows, annot=True, fmt='.2%', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 8},ax =axs[i,j])
                  axs[i,j].set_title('+'.join(self.data[m]['features']) + '\n' + m, fontsize = 7, color='red')
                  axs[i,j].set_xlabel('Predicted Labels')
                  axs[i,j].set_ylabel('True Labels')
                except Exception as e:
                  print("continue")
            
           
        
        fig.tight_layout(pad=5)
        
        fig.suptitle('Confusion matrix'+ '  ' + self.title, x=0.5, y=0.99)
        plt.savefig(self.output_directory + self.title + '_confusion_matrix.pdf', format='pdf',dpi=300, bbox_inches='tight')
       # plt.show()
        
        
    '''
     input:
    output: fscore bar plot
    '''
    def plot_f1score(self, hormones_list,models_list):
        num_keys = len(hormones_list)
        num_items = len(models_list)
        
        fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
        for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
       
        i=0
        for j, m in enumerate(models_list):
              try:
                list_fscore = self.data[m]['fscore']   
                array_mean =100*np.mean(list_fscore, axis =0) 
                array_se = 100*np.std(list_fscore, axis=0, ddof=1)/np.sqrt(len(list_fscore))
                array_mean_list = array_mean.tolist()
                array_se_list = array_se.tolist()
                #get classes
                class_names = self.data[m]['classes']
                axs[i,j].set_axis_on()
                #bar plot
                axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("f-score % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
              except Exception as e:
                  print("continue")
              
        fig.tight_layout(pad=5)
        fig.suptitle('f-score: 2X(precxrecall)/(prec+recall)'+ '  ' + self.title, x=0.5, y=0.99) 
        plt.savefig(self.output_directory + self.title + '_fscore' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
      # Show the plot
      #plt.show()
                
   
    '''
    input:
    output: plot important features
    '''   
    def plot_important_features(self, hormones_list,models_list):
        num_keys = len(hormones_list)
        num_items = len(models_list)
        
        fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
        for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
       
        i=0
        for j, m in enumerate(models_list):
              try:
                list_mean = self.data[m]['mean_abs_shap_values']   
                array_mean =np.mean(list_mean, axis =0) 
                array_se = np.std(list_mean, axis=0, ddof=1)/np.sqrt(len(list_mean))
                categories = self.data[m]['features']
                sorted_indices = np.argsort(array_mean)[::1]
                array_mean = array_mean[sorted_indices]
                array_se = array_se[sorted_indices]
                categories =[categories[i] for i in sorted_indices]
                
                axs[i,j].set_axis_on()
                #bar plot
                axs[i,j].barh(categories, array_mean, yerr = array_se, capsize = 10)
                axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_xlabel("mean(abs(shap.values))", fontsize = 8) 
                axs[i,j].yticks(fontsize=6)

              except Exception as e:
                  print("continue")
        plt.savefig(self.output_directory + self.title + '_important_features' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')   
                
    '''
       input: dictionary
       output: dataframe
      ''' 
    def create_table(self):
       # Create a dictionary where values are empty lists by default
       data_dict = defaultdict(list)
       
       
       for m, inner_dict in self.data.items():
                 data_dict['sex'] = self.sex
                 class1 = str(inner_dict['classes'][0])
                 class2 = str(inner_dict['classes'][1])
                 data_dict['class 1'].append(str(inner_dict['classes'][0]))
                 data_dict['class 2'].append(str(inner_dict['classes'][1]))
                 data_dict['hormones'].append(tuple(inner_dict['features']))
                 data_dict['models'].append(m)                  
                 data_dict['confusion_matrix'].append(inner_dict['confusion_matrix'])
                 data_dict['fscore_' + 'class 1'].append((100*(inner_dict['fscore'] ))[0])
                 data_dict['fscore_' + 'class 2'].append((100*(inner_dict['fscore'] ))[1])
                 
               
                 data_dict['accuracy'].append(100*(inner_dict['accuracy'] ))
                 data_dict['balanced_accuracy'].append(100*(inner_dict['balanced_accuracy'] ))
                # data_dict['se_accuracy'].append(100*np.std(inner_dict['accuracy'], axis=0, ddof=1)/np.sqrt(len(inner_dict['accuracy'])))
                 data_dict['roc_auc'].append(100*(inner_dict['roc_auc'] ))
                
                 data_dict['precision_' + 'class 1'].append((100*(inner_dict['precision']))[0])
               #  data_dict['se_precision_' + class1].append((100*np.std(inner_dict['precision'], axis=0, ddof=1)/np.sqrt(len(inner_dict['precision'])))[0])
                 
                 data_dict['precision_' + 'class 2'].append((100*(inner_dict['precision'] ))[1])
                # data_dict['se_precision_' + class2].append((100*np.std(inner_dict['precision'], axis=0, ddof=1)/np.sqrt(len(inner_dict['precision'])))[1])
                 
                 data_dict['recall_' + 'class 1'].append((100*(inner_dict['recall'] ))[0])
                 #data_dict['se_recall_' + class1].append((100*np.std(inner_dict['recall'], axis=0, ddof=1)/np.sqrt(len(inner_dict['recall'])))[0])
                 
                 data_dict['recall_' +'class 2'].append((100*(inner_dict['recall'] ))[1])
                 #data_dict['se_recall_' + class2].append((100*np.std(inner_dict['recall'], axis=0, ddof=1)/np.sqrt(len(inner_dict['recall'])))[1])
                 
                
                # data_dict['se_fscore_' + class1].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[0])
                 
                
                 #data_dict['se_fscore_' + class2].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[1])
                 
                 #confusion matrix
                 sum_cm = np.sum(inner_dict['confusion_matrix'], axis=1)
                 
                 data_dict['% true values' +  'class 1'].append(inner_dict['confusion_matrix'][0,0]*100/sum_cm[0])
                 data_dict['% true values' + 'class 2'].append(inner_dict['confusion_matrix'][1,1]*100/sum_cm[1])
                 
                 
                 
                 
                         
        #  Convert into dataframe
        
       total_data = pd.DataFrame(data_dict)
       return   total_data,data_dict
    
    '''
       input: dictionary
       output: dataframe
      ''' 
    def create_table_before(self):
       # Create a dictionary where values are empty lists by default
       data_dict = defaultdict(list)
       
       
       for m, inner_dict in self.data.items():
                 data_dict['sex'] = self.sex
                 class1 = str(inner_dict['classes'][0])
                 class2 = str(inner_dict['classes'][1])
                 data_dict['class 1'].append(str(inner_dict['classes'][0]))
                 data_dict['class 2'].append(str(inner_dict['classes'][1]))
                 data_dict['hormones'].append(tuple(inner_dict['features_before']))
                 data_dict['models'].append(m)                  
                 data_dict['confusion_matrix'].append(inner_dict['confusion_matrix_b'])
                 data_dict['fscore_' + 'class 1'].append((100*(inner_dict['fscore_b'] ))[0])
                 data_dict['fscore_' + 'class 2'].append((100*(inner_dict['fscore_b'] ))[1])
                 
               
                 data_dict['accuracy'].append(100*(inner_dict['accuracy_b'] ))
                 data_dict['balanced_accuracy'].append(100*(inner_dict['balanced_accuracy_b'] ))
                # data_dict['se_accuracy'].append(100*np.std(inner_dict['accuracy'], axis=0, ddof=1)/np.sqrt(len(inner_dict['accuracy'])))
                 data_dict['roc_auc'].append(100*(inner_dict['roc_auc_b'] ))
                
                 data_dict['precision_' + 'class 1'].append((100*(inner_dict['precision_b']))[0])
               #  data_dict['se_precision_' + class1].append((100*np.std(inner_dict['precision'], axis=0, ddof=1)/np.sqrt(len(inner_dict['precision'])))[0])
                 
                 data_dict['precision_' + 'class 2'].append((100*(inner_dict['precision_b'] ))[1])
                # data_dict['se_precision_' + class2].append((100*np.std(inner_dict['precision'], axis=0, ddof=1)/np.sqrt(len(inner_dict['precision'])))[1])
                 
                 data_dict['recall_' + 'class 1'].append((100*(inner_dict['recall_b'] ))[0])
                 #data_dict['se_recall_' + class1].append((100*np.std(inner_dict['recall'], axis=0, ddof=1)/np.sqrt(len(inner_dict['recall'])))[0])
                 
                 data_dict['recall_' +'class 2'].append((100*(inner_dict['recall_b'] ))[1])
                 #data_dict['se_recall_' + class2].append((100*np.std(inner_dict['recall'], axis=0, ddof=1)/np.sqrt(len(inner_dict['recall'])))[1])
                 
                
                # data_dict['se_fscore_' + class1].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[0])
                 
                
                 #data_dict['se_fscore_' + class2].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[1])
                 
                 #confusion matrix
                 sum_cm = np.sum(inner_dict['confusion_matrix_b'], axis=1)
                 
                 data_dict['% true values' +  'class 1'].append(inner_dict['confusion_matrix_b'][0,0]*100/sum_cm[0])
                 data_dict['% true values' + 'class 2'].append(inner_dict['confusion_matrix_b'][1,1]*100/sum_cm[1])
                 
                 
                 
                 
                         
        #  Convert into dataframe
        
       total_data = pd.DataFrame(data_dict)
       return   total_data,data_dict
    
    
    '''
    input: total filter
    output: only data with significant confusion values
    '''
    def  filter_confusion(self,data,class1,class2):
        filter_data =data[(data['% true values' +  class1] > 60) & (data['% true values' +  class2] > 60)]
        return filter_data
                
    '''
    input: total filter
    output: only data with significant precision values
    ''' 
    def filter_precision(self,data, class1,class2):
       filter_data = data[(data[ 'precision_' + class1] > 50) & (data[ 'precision_' + class2] > 50)
                          & (data[ 'recall_' + class1] > 50) & (data[ 'recall_' + class2] > 50)] 
       return filter_data
   
    '''
    input: total filter
    output: create dictionary each hormone and the respective models
    '''
    def create_dict(self,data):
        # Create dictionary
        hormone_dict = data.groupby('hormones')['models'].apply(list).to_dict()
        return hormone_dict
    
        
    '''
    input: nested dictionary
    output : get the keys of all the dictionaries
      note: recursive method is used
    '''
    @staticmethod
    def separate_keys(d, level=0, result=None):
        if result is None:
            result = {}  # Initialize the result dictionary
        
        # Ensure a list exists for the current level
        if level not in result:
            result[level] = []
        
        for key, value in d.items():
            result[level].append(key)  # Add the current key to the corresponding level
            if isinstance(value, dict):  # If the value is a dictionary, recurse
                plot_data.separate_keys(value, level + 1, result)
        
        return result
    '''
    plot important features
    '''
    def PlotFeaturesShap(self,hormones_list,models_list):
      num_keys = len(hormones_list)
      num_items = len(models_list)
        
      fig, axs = plt.subplots((num_keys+1) , (num_items +1),  figsize=((num_items+1)*6, (num_keys+1)*4)) #good 60
      fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
      for i in range((num_keys+1)):
           for j in range((num_items+1)):
               axs[i,j].set_axis_off()
       
      i=0
      for j, m in enumerate(models_list):
         
          shap_values = self.data[m]['shap_values']
          categories = self.data[m]['features']
          sorted_indices = np.argsort(shap_values)[::1]
          shap_values_sort = shap_values[sorted_indices]
          
          categories =[categories[i] for i in sorted_indices]
          
          axs[i,j].set_axis_on()
          axs[i,j].barh(categories, shap_values_sort)
          axs[i,j].set_title(("+".join(self.data[m]['features'])) + '\n' + m, fontsize = 7, color='red')
          axs[i, j].set_xlabel("abs(shap.values)", fontsize = 8) 
          #axs[i,j].yticks(fontsize=6)
      plt.savefig(self.output_directory + self.title + '_important_features' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')   
      
      
    '''
    plot shap values
    '''
    def plot_trad_shap(self):
        fig, axs = plt.subplots(2 , 3,  figsize=(200, 100)) 
        #plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3) #good 60

       # fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots
        for i in range(2):
           for j in range(3):
               axs[i,j].set_axis_off()


        for m, inner_dict in self.data.items():
         fig.suptitle(tuple(inner_dict['classes']), fontsize=10)
         all_shap_values = inner_dict['shap_values']
         # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
         all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
         X = inner_dict['data_features']
         feature_names = X.columns
         
         #only consider the 10 main features according to absolute value
         mean_abs_shap = np.abs(all_shap_values_1).mean(axis=0)
         top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
         X_top = X.iloc[:,top_indices]
         shap_values_top =all_shap_values_1[:,top_indices]
         
         
        # Plot 1: Traditional SHAP Summary (Beeswarm) Plot
         #plt.figure(figsize=(12, 12)) 
         plt.sca(axs[0,2])
         axs[0,2].set_axis_on()
         shap.summary_plot(shap_values_top, X_top , feature_names=X_top.columns.tolist(),plot_type='violin',show=False)
         axs[0,2].set_title('Feature influence on prediction',fontsize=8)
         axs[0,2].set_xlabel("SHAP value \n (impact on model output)")
         axs[0,2].xaxis.label.set_size(6)
         axs[0,2].yaxis.label.set_size(6)
         for tick in axs[0,2].get_yticklabels():
             tick.set_fontsize(6)
         for tick in axs[0,2].get_xticklabels():
             tick.set_fontsize(6)
         axs[0,2].set_xlim(-2.5,2.5)
         subdivisions = [-2.5,0,2.5]
         axs[0,2].set_xticks(subdivisions)
         # Remove all PathCollection (scatter plots) from axes
         for collection in list(axs[0,2].collections):
             if isinstance(collection, mcoll.PathCollection):
                 collection.remove()
         
         
         
         #plt.figure(figsize=(12, 12)) 
         plt.sca(axs[0,1])
         axs[0,1].set_axis_on()
         shap.summary_plot(shap_values_top, X_top , feature_names=X_top.columns.tolist(),show=False)
         axs[0,1].set_title('Feature influence on prediction',fontsize=8)
         axs[0,1].set_xlabel("SHAP value \n (impact on model output)")
         axs[0,1].xaxis.label.set_size(6)
         axs[0,1].yaxis.label.set_size(6)
         for tick in axs[0,1].get_yticklabels():
             tick.set_fontsize(6)
         for tick in axs[0,1].get_xticklabels():
             tick.set_fontsize(6)
         axs[0,1].set_xlim(-2.5,2.5)
         subdivisions = [-2.5,0,2.5]
         axs[0,1].set_xticks(subdivisions)
         
                  
         # Plot 2: SHAP Summary Bar Plot
         #plt.figure(figsize=(12, 12)) 
         plt.sca(axs[0,0])
         axs[0,0].set_axis_on()
         shap.summary_plot(shap_values_top, X_top , feature_names=X_top.columns.tolist(), plot_type="bar", show =False)
         axs[0,0].set_title('Ten important features',fontsize=8)
         axs[0,0].set_xlabel("mean(|SHAP value|) \n (average impact on model output)")
         axs[0,0].xaxis.label.set_size(6)
         axs[0,0].yaxis.label.set_size(6)
         axs[0,0].set_xlim(0,1.5)
         subdivisions = [0,0.5,1,1.5]  # every 0.5 units
         axs[0,0].set_xticks(subdivisions)
         for tick in axs[0,0].get_yticklabels():
             tick.set_fontsize(6)
         for tick in axs[0,0].get_xticklabels():
             tick.set_fontsize(6)
         
         
         
         class_ = inner_dict['classes']
         plt.sca(axs[1,0]) 
         axs[1,0].set_axis_on()
         fscore_actual = (inner_dict['fscore'])[0]
         pval = np.mean(inner_dict['shuffle_fscore_class1'] >= fscore_actual)
         plt.hist(inner_dict['shuffle_fscore_class1'], bins=6, alpha=0.7, color='blue', edgecolor='black')
         plt.axvline(fscore_actual, color='red',linestyle='--', label= f"Actual F-score = {fscore_actual}")
         plt.text(fscore_actual + 0.1, plt.ylim()[1]*0.8, f"pvalue = {pval:.3f}", color='red',fontsize = 6)
         axs[1,0].set_title('random permutation',fontsize=8)
         axs[1,0].set_xlim(0,1)
         axs[1,0].set_ylabel("Frequency")
         axs[1,0].set_xlabel('shuffle_fscore_' + class_[0])
         axs[1,0].xaxis.label.set_size(6)
         axs[1,0].yaxis.label.set_size(6)
         for tick in axs[1,0].get_yticklabels():
             tick.set_fontsize(6)
         for tick in axs[1,0].get_xticklabels():
             tick.set_fontsize(6)
         
         
         plt.sca(axs[1,1]) 
         axs[1,1].set_axis_on()
         fscore_actual = (inner_dict['fscore'])[1]
         pval = np.mean(inner_dict['shuffle_fscore_class2'] >= fscore_actual)
         plt.hist(inner_dict['shuffle_fscore_class2'], bins=6, alpha=0.7, color='blue', edgecolor='black')
         plt.axvline(fscore_actual, color='red',linestyle='--', label= f"Actual F-score = {fscore_actual}")
         plt.text(fscore_actual + 0.1, plt.ylim()[1]*0.8, f"pvalue = {pval:.3f}", color='red',fontsize = 6)
         axs[1,1].set_title('random permutation',fontsize = 8)
         axs[1,1].set_xlim(0,1)
         axs[1,1].set_ylabel("Frequency")
         axs[1,1].set_xlabel('shuffle_fscore_' + class_[1])
         axs[1,1].xaxis.label.set_size(6)
         axs[1,1].yaxis.label.set_size(6)
         for tick in axs[1,1].get_yticklabels():
             tick.set_fontsize(6)
         for tick in axs[1,1].get_xticklabels():
             tick.set_fontsize(6)
             
        #add confusion matrix
        plt.sca(axs[1,2]) 
        axs[1,2].set_axis_on()
        cm = inner_dict['confusion_matrix']
        class_names = inner_dict['classes']
        sum_cm = np.sum(inner_dict['confusion_matrix'], axis=1)
        #Normalize by the total number of instances per class
        sns.heatmap(cm/sum_cm, annot=True, fmt='.2%', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 8},ax =axs[i,j])
        axs[1,2].set_title(('confusion matrix') , fontsize = 8)
        axs[1,2].set_xlabel('Predicted Labels',fontsize = 6)
        axs[1,2].set_ylabel('True Labels', fontsize = 6)
        for tick in axs[1,2].get_yticklabels():
             tick.set_fontsize(6)
        for tick in axs[1,2].get_xticklabels():
             tick.set_fontsize(6)
        
        plt.tight_layout()
        
        # # Get the figure manager
        # manager = plt.get_current_fig_manager()
        # manager.window.state('zoomed')  
        
        # plt.ion()
        # plt.show()
        # plt.pause(0.1)
        # plt.gca()
        plt.savefig(self.output_directory + self.title + '_important_features' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')  
       # plt.close() 
        
        
        
        
        a=1
      
      
    ##############################################
    '''
    input: shap values
    output: shap values for each sex, including mean of absolute value including standard error
    '''
    def getShap(self, sex):
        sex_indexes = []
        for m, inner_dict in self.data.items():
          #get shap values
          all_shap_values = inner_dict['shap_values']
         # Concatenate SHAP values from all folds (shape: (n_samples, n_features))
          all_shap_values_1 = np.concatenate(all_shap_values, axis=0)
          #get maximo abs value
          max_abs_value = np.max(np.abs(all_shap_values_1))
          
          #get features
          X = inner_dict['data_features']
          feature_names = X.columns
          if (sex == 'male') or (sex == 'female'):
             sex_shap = inner_dict['values_sex_test']
             sex_values = [s.iloc[0] for s in sex_shap]
             sex_indexes = [i for i,sexs in enumerate(sex_values) if sexs == sex]
             shap_sex = all_shap_values_1[sex_indexes,:]
             original_data = X.iloc[sex_indexes,:]
          else:
             shap_sex = all_shap_values_1
             original_data = X
         #only consider the 10 main features according to absolute value
          #calculate mean
          mean_abs_shap = np.abs(shap_sex).mean(axis=0)
          #calculate standard error
          abs_data = np.abs(shap_sex)
          stderror_abs_shap = np.std(abs_data, axis = 0, ddof = 1)/np.sqrt(abs_data.shape[0])
          
          #order in a given order
          #top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
          top_indices = np.argsort(mean_abs_shap)[::-1]
          X_top = X.iloc[:,top_indices]
          shap_values_top =mean_abs_shap[top_indices]
          shap_values_error = stderror_abs_shap[top_indices]
          categories = X_top.columns.tolist()
          
          classes = inner_dict['classes']
          
          
          return categories, shap_values_top, shap_values_error,classes,max_abs_value,shap_sex,original_data, sex_indexes
          
          
          
         
          
    '''
        Input: data to plot
        Output bar plot
        '''
    
    def plotBarPlot(self,categories,shap_values_top,shap_values_error,ax,sex,maximo):
           fig, ax_alone =plt.subplots(figsize=(8, 5))
           for axs in [ax, ax_alone]: 
            axs.barh(y=categories, width=shap_values_top, xerr=shap_values_error, capsize=3, color='royalblue', edgecolor='none')
            axs.invert_yaxis() 
            axs.set_title(sex)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['left'].set_color('gray')
            axs.set_xlabel("mean(|SHAP value|")
            ax.set_yticklabels(categories,fontsize = 6)
            #remove y axis ticks
            
            axs.set_xlim(0,1.6)
            
            
            if axs == ax_alone:
               plt.savefig(self.output_directory + self.title + '_important_features_1' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')  
               plt.close()
           #ax.gca().invert_yaxis()
            
           
    def arrange(self,all_categories,all_shap_abs,all_shap_abs_err):  
         
        #  indices_males = [(all_categories[1]).index(item) for item in (all_categories[0])[0:9]]
        #  indices_females = [(all_categories[2]).index(item) for item in (all_categories[0])[0:9]]
         
         indices_males = [(all_categories[1]).index(item) for item in (all_categories[0])]
         indices_females = [(all_categories[2]).index(item) for item in (all_categories[0])]
         all_shap_abs_males = [(all_shap_abs[1])[i] for i in indices_males ]
         all_shap_abs_females = [(all_shap_abs[2])[i] for i in indices_females ]
         all_shap_abs_err_males = [(all_shap_abs_err[1])[i] for i in indices_males ]
         all_shap_abs_err_females = [(all_shap_abs_err[2])[i] for i in indices_females ]
         
         return all_shap_abs_males,all_shap_abs_females, all_shap_abs_err_males, all_shap_abs_err_females
    '''
    input:data
    '''  
    def twobarplot(self,labels,abs_males,abs_females, err_males, err_females,ax ):
      y_pos = np.arange(len(labels))*2
      bar_width = 0.7
      ax.barh(y_pos - bar_width/2, abs_males, xerr=err_males, height=bar_width,
        label='Male', color='skyblue', capsize=1,edgecolor='none',error_kw=dict(elinewidth=1, capsize=1, capthick=1))
      ax.barh(y_pos + bar_width/2, abs_females, xerr=err_females, height=bar_width,
        label='Female', color='lightpink', capsize=1,edgecolor='none',error_kw=dict(elinewidth=1, capsize=1, capthick=1))
      
      ax.invert_yaxis() 
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['left'].set_color('gray')
      ax.set_yticks(y_pos)
      ax.set_yticklabels(labels,fontsize = 6)
      ax.set_xlabel("mean(|SHAP value|")
      

      ax.legend()
      
    '''
      input:shap values
      output: shap values were ordered
      '''
    def arrange_values(self,all_shap_values,original_features,order_categories,all_original_features):
        #find index of the order category
        indices= [(original_features.columns).get_loc(item) for item in order_categories]
        all_shap_values_ordered = all_shap_values[:,indices]
        all_original_features_ordered = all_original_features.iloc[:,indices]
        return all_shap_values_ordered,  all_original_features_ordered
    
    '''
    output: shap plots
    '''
    def plotDotPlot(self,all_shap_values_ordered,all_original_features_ordered,ordered_categories, sex,ax,indices_sex): 
           fig, ax_alone =plt.subplots(figsize=(8, 5))
           # Create a smooth gradient from blue → violet → red
           blue_violet_red = LinearSegmentedColormap.from_list(
                'blue_violet_red',
              ['blue', 'violet', 'red']
              )
           for axs in [ax, ax_alone]: 
              n_total_samples, n_features = all_shap_values_ordered.shape
              for i in range(n_features):
                  vals = all_shap_values_ordered[:, i]
                  # Use feature value to color it use allfeatures both male and emale
                  raw_color = all_original_features_ordered.iloc[:, i].values.reshape(-1,1)
                  # Normalize to 0–1
                  color = MinMaxScaler().fit_transform(raw_color).flatten()
                  
                  if (sex=='male') or (sex == 'female'):
                    color = color[indices_sex] #choose the given colors
                  
                  # Jitter y-axis
                  
                  y = np.random.normal(i, 0.1, size=vals.shape[0])
                  scatter = axs.scatter(vals, y, c=color, cmap= blue_violet_red , s=40, alpha=1, edgecolor='k', linewidth=0.2)
                  axs.hlines(i, xmin=-2.5, xmax=2.5,color='lightgray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
              # Label formatting
              axs.set_yticks(range(n_features))
              axs.set_yticklabels(ordered_categories,fontsize =8)
              axs.axvline(0, linestyle='-', color='gray', linewidth=1)
              axs.set_xlabel('SHAP value')
              axs.set_title(sex)
              axs.invert_yaxis()  # To match SHAP default
              axs.set_xlim(-2.5,2.5)
              
              cbar = fig.colorbar(scatter, ax=axs,aspect=40, pad=0.01)
              cbar.set_label('Feature value')
              cbar.ax.tick_params(size=0, labelsize=0)  # Hides ticks and numbers
              cbar.ax.text(0.5, -0.05, 'Low', ha='center', va='top', transform=cbar.ax.transAxes)
              cbar.ax.text(0.5, 1.05, 'High', ha='center', va='bottom', transform=cbar.ax.transAxes)

              
              if axs == ax_alone:
               plt.savefig(self.output_directory + self.title + '_important_features_1' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')  
               plt.close()
               
      
    '''
    output: shap plots- by marking the sex of species
    '''
    def plotDot_with_Sex(self,all_shap_values_ordered,all_original_features_ordered,ordered_categories, sex,ax,indices_sex): 
      
           fig, ax_alone =plt.subplots(figsize=(8, 5))
           # Create a smooth gradient from blue → violet → red
           blue_violet_red = LinearSegmentedColormap.from_list(
                'blue_violet_red',
              ['blue', 'violet', 'red']
              )
           for axs in [ax, ax_alone]: 
              n_total_samples, n_features = all_shap_values_ordered.shape
              for i in range(n_features):
                vals = all_shap_values_ordered[:, i]
                  # Use feature value to color it use allfeatures both male and emale
                raw_color = all_original_features_ordered.iloc[:, i].values.reshape(-1,1)
                  # Normalize to 0–1
                color = MinMaxScaler().fit_transform(raw_color).flatten()
                  
                
                  #separate females and males
                  #males
                vals_s = vals[indices_sex[1]]
                y_s = np.random.normal(i, 0.1, size=vals_s.shape[0])
                color_s = color[indices_sex[1]] #choose the given colors
                  # Jitter y-axis
                scatter = axs.scatter(vals_s, y_s, c=color_s, cmap= blue_violet_red , s=40, alpha=1, edgecolor='k', linewidth=0.2,marker = 'o')
                  
                  #females
                vals_f = vals[indices_sex[2]]
                y_f = np.random.normal(i, 0.1, size=vals_f.shape[0])
                color_f = color[indices_sex[2]] #choose the given colors
                  # Jitter y-axis
                scatter = axs.scatter(vals_f, y_f, c=color_f, cmap= blue_violet_red , s=40, alpha=1, edgecolor='k', linewidth=0.2, marker = 's')
 
                axs.hlines(i, xmin=-2.5, xmax=2.5,color='lightgray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
              # Label formatting
              axs.set_yticks(range(n_features))
              axs.set_yticklabels(ordered_categories,fontsize =8)
              axs.axvline(0, linestyle='-', color='gray', linewidth=1)
              axs.set_xlabel('SHAP value')
              axs.set_title("all")
              axs.invert_yaxis()  # To match SHAP default
              axs.set_xlim(-2.5,2.5)
              
              cbar = fig.colorbar(scatter, ax=axs,aspect=40, pad=0.01)
              cbar.set_label('Feature value')
              cbar.ax.tick_params(size=0, labelsize=0)  # Hides ticks and numbers
              cbar.ax.text(0.5, -0.05, 'Low', ha='center', va='top', transform=cbar.ax.transAxes)
              cbar.ax.text(0.5, 1.05, 'High', ha='center', va='bottom', transform=cbar.ax.transAxes)
              
              # Create custom legend handles
              legend_elements = [
                  Line2D([0], [0], marker='o', color='w', label='Male',
                        markerfacecolor='black', markersize=10),
                  Line2D([0], [0], marker='s', color='w', label='Female',
                        markerfacecolor='black', markersize=10)
              ]

              # Add the legend with the custom handles
              axs.legend(handles=legend_elements, loc='lower right')

              
              if axs == ax_alone:
               plt.savefig(self.output_directory + self.title + '_important_features_1' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')  
               plt.close()
    
           
    def __call__(self,select_column_prob):
        all_categories = []
        all_shap_abs = []
        all_shap_abs_err = []
        all_shap_values = []
        all_original_features = []
        all_sex_indices = []
        df_all_shape_values = pd.DataFrame()  
        df_all_abs_shape_values = pd.DataFrame() 
        
        total_data, data_dict = self.create_table()
       
        #list hormones
        hormones_list = list(set(data_dict['hormones']))
        models_list = list(set(data_dict['models']))
       
        
        #do this for significant F score
        fscore_list = self.data[models_list[0]]['fscore']
        ####################################################### bar plot graph
        fig , axs = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(4):
               axs[i // 2, i % 2].set_axis_off()
        if (fscore_list[0] >= 0.6) and (fscore_list[1] >= 0.6):
             for i,sex in enumerate(['all','male','female']): 
                axs[i // 2,i % 2].set_axis_on() 
                categories, shap_values_top, shap_values_error,classes,maximo,shap_sex_allvalues,original_features, sex_indexes = self.getShap(sex)
                self.plotBarPlot(categories, shap_values_top, shap_values_error,axs[i // 2,i % 2],sex,np.ceil(maximo))
                all_categories.append(categories)
                all_shap_abs.append(shap_values_top)
                all_shap_abs_err.append(shap_values_error)
                all_shap_values.append(shap_sex_allvalues)
                all_original_features.append(original_features)
                all_sex_indices.append(sex_indexes)
              
             all_shap_abs_males,all_shap_abs_females, all_shap_abs_err_males, all_shap_abs_err_females = self.arrange(all_categories,all_shap_abs,all_shap_abs_err) 
             axs[1,1].set_axis_on() 
             #self.twobarplot((all_categories[0])[0:9],all_shap_abs_males,all_shap_abs_females, all_shap_abs_err_males, all_shap_abs_err_females,axs[1,1] )  
             self.twobarplot((all_categories[0]),all_shap_abs_males,all_shap_abs_females, all_shap_abs_err_males, all_shap_abs_err_females,axs[1,1] )  
             fig.suptitle(tuple(classes), fontsize=12)
             
             plt.tight_layout()
             plt.savefig(self.output_directory + self.title + '_important_features_full' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')
             ################################################
              ####################################################### dots plot graph
        fig , axs = plt.subplots(2, 2, figsize=(16, 12))
        for i in range(4):
                    axs[i // 2, i % 2].set_axis_off()
        if (fscore_list[0] >= 0.6) and (fscore_list[1] >= 0.6):
                  for i,sex in enumerate(['all','male','female']): 
                      axs[i // 2,i % 2].set_axis_on() 
                      all_shap_values_ordered , all_original_features_ordered= self.arrange_values(all_shap_values[i],original_features,(all_categories[0]),all_original_features[0]) #take all the features
                      #for all
                      if sex == 'all':
                          all_shap_values_ordered_for_all = all_shap_values_ordered
                          all_original_features_ordered_for_all = all_original_features_ordered
                          
                      self.plotDotPlot(all_shap_values_ordered,all_original_features_ordered,(all_categories[0]), sex,axs[i // 2,i % 2],all_sex_indices[i])
                  axs[1,1].set_axis_on() 
                  all_shap_values_ordered , all_original_features_ordered= self.arrange_values(all_shap_values[0],original_features,(all_categories[0]),all_original_features[0]) #take all the features    
                  self.plotDot_with_Sex(all_shap_values_ordered,all_original_features_ordered,(all_categories[0]), sex,axs[1,1],all_sex_indices)  
                 
                    
                  fig.suptitle(tuple(classes), fontsize=12)
                  
                  plt.tight_layout()
                  plt.savefig(self.output_directory + self.title + '_shap_values_all' +'.pdf', format='pdf',dpi=300,bbox_inches='tight')
            #  ################################################
        # save shap values in an excel file
       # self.saveShap()   
             
                  df_all_shape_values = pd.DataFrame(all_shap_values_ordered, columns = all_original_features_ordered.columns.to_list())
                  df_all_features_values =  pd.DataFrame(all_original_features_ordered)
                  abs_arrays = [np.abs(arr) for arr in all_shap_values_ordered]
                  df_all_abs_shape_values = pd.DataFrame(abs_arrays, columns = all_original_features_ordered.columns.to_list())
                  
                  df_all_shape_values['sex'] = np.nan
                  df_all_features_values['sex'] = np.nan
                  df_all_abs_shape_values['sex'] = np.nan
                  
                  df_all_shape_values.loc[all_sex_indices[1],'sex'] = 'male'
                  df_all_shape_values.loc[all_sex_indices[2],'sex'] = 'female'
                  
                  # df_all_features_values.loc[all_sex_indices[1],'sex'] = 'male'
                  # df_all_features_values.loc[all_sex_indices[2],'sex'] = 'female'
                  
                  df_all_abs_shape_values.loc[all_sex_indices[1],'sex'] = 'male'
                  df_all_abs_shape_values.loc[all_sex_indices[2],'sex'] = 'female'
                  
             
             #plt.show() 
             
                
                
   
            
       

        return total_data, df_all_shape_values,  df_all_abs_shape_values
     