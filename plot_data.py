import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
#matplotlib.use('TkAgg') 
#plt.ion() # Turn on the interactive mode


class plot_data:
    def __init__(self,data,title,output_directory):
        self.data = data
        self.title = title
        self.output_directory = output_directory
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
    def plot_precision(self, hormone_dict):
       num_keys = len(hormone_dict)
       num_items = [len(v) for v in hormone_dict.values()]
       fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1),  figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60
       fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
       for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
       
       for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_precision = self.data[h][m]['precision']   
                array_mean =100*np.mean(list_precision, axis =0) 
                array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                array_mean_list = array_mean.tolist()
                array_se_list = array_se.tolist()
                #get classes
                class_names = self.data[h][m]['classes']
                #bar plot
                axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Precision % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
        
       fig.tight_layout(pad=5)
       fig.suptitle('Precision: TP/(TP+FN)'+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_precision' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       #plt.show()
            
    '''
    input:
    output: accuracy bar plot
    '''
    def plot_accuracy(self, hormone_dict):
       num_keys = len(hormone_dict)
       num_items = [len(v) for v in hormone_dict.values()]
       fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1),  figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60
       fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

       # turn off the axes
       for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
               
       for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_precision = self.data[h][m]['accuracy']   
                array_mean =100*np.mean(list_precision) 
                array_mean_list = array_mean.tolist()
                #get classes
                class_names = self.data[h][m]['classes']
                #bar plot
                axs[i,j].bar("two_classes", array_mean.tolist(), color=['skyblue'], edgecolor = 'black',width = 0.01)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Accuracy % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
          
       fig.tight_layout(pad=5)
       fig.suptitle('Accuracy: '+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_accuracy' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       #plt.show()
            
    '''
    input:
    output: recagg bar plot
    '''
    def plot_recall(self, hormone_dict):
      num_keys = len(hormone_dict)
      num_items = [len(v) for v in hormone_dict.values()]
      fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1),  figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60
      fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
      for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
       
      for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_recall = self.data[h][m]['recall']   
                array_mean =100*np.mean(list_recall, axis =0) 
                array_se = 100*np.std(list_recall , axis=0, ddof=1)/np.sqrt(len(list_recall ))
                array_mean_list = array_mean.tolist()
                array_se_list = array_se.tolist()
                
                #get classes
                class_names = self.data[h][m]['classes']
                #bar plot
                axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Recall % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
       
      fig.tight_layout(pad=5)
      fig.suptitle('Recall: TP/(TP+FP)'+ '  ' + self.title, x=0.5, y=0.99) 
      plt.savefig(self.output_directory + self.title + '_recall' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
      #plt.show()
    '''
    input:
    output: plot confusion matrix
    '''
    def plot_confusion_matrix(self, hormone_dict):
        num_keys = len(hormone_dict)
        num_items = [len(v) for v in hormone_dict.values()]
        
        fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1),  figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60
        fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
        for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
        
        #fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 15))
        
        
        for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_confusion_matrix = self.data[h][m]['confusion_matrix']
                #get classes
                class_names = self.data[h][m]['classes']
                # Calculate the sum of all the confusion matrix
                sum_cm = np.sum(list_confusion_matrix, axis=0)
                #calculate the sum of the rows
                sum_rows = sum_cm.sum(axis=1, keepdims=True) 
                #Normalize by the total number of instances per class
                sns.heatmap(sum_cm/sum_rows, annot=True, fmt='.2%', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 8},ax =axs[i,j])
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i,j].set_xlabel('Predicted Labels')
                axs[i,j].set_ylabel('True Labels')
            
           
        
        fig.tight_layout(pad=5)
        
        fig.suptitle('Confusion matrix'+ '  ' + self.title, x=0.5, y=0.99)
        plt.savefig(self.output_directory + self.title + '_confusion_matrix.pdf', format='pdf',dpi=300, bbox_inches='tight')
       # plt.show()
        
        
    '''
     input:
    output: fscore bar plot
    '''
    def plot_f1score(self, hormone_dict):
      num_keys = len(hormone_dict)
      num_items = [len(v) for v in hormone_dict.values()]
      fig, axs = plt.subplots((num_keys+1) , (max(num_items)+1),  figsize=((max(num_items)+1)*6, (num_keys+1)*4)) #good 60
      fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust space between subplots

        # turn off the axes
      for i in range((num_keys+1)):
           for j in range((max(num_items)+1)):
               axs[i,j].set_axis_off()
       
      for i, (h, list_models) in enumerate(hormone_dict.items()):
              for j, m in enumerate(list_models):
                axs[i,j].set_axis_on()
                list_fscore = self.data[h][m]['fscore']   
                array_mean =100*np.mean(list_fscore, axis =0) 
                array_se = 100*np.std(list_fscore, axis=0, ddof=1)/np.sqrt(len(list_fscore))
                array_mean_list = array_mean.tolist()
                array_se_list = array_se.tolist()
                #get classes
                class_names = self.data[h][m]['classes']
                #bar plot
                axs[i,j].bar(class_names, array_mean.tolist(), yerr = array_se_list, capsize = 5, color=['skyblue', 'salmon'], edgecolor = 'black', alpha = 0.7)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("f-score % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 120)
              
      fig.tight_layout(pad=5)
      fig.suptitle('f-score: 2X(precxrecall)/(prec+recall)'+ '  ' + self.title, x=0.5, y=0.99) 
      plt.savefig(self.output_directory + self.title + '_fscore' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
      # Show the plot
      #plt.show()
                
    '''
    input:
    output: prob histogram two distribution random and exactly of accuracy
    ''' 
    def plot_boot_histograms(self, hormones,models):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
      
       i = 0
       for h in hormones:
            j = 0
            for m in models:
                data_exactly = self.data[h][m]['accuracy'] 
                data_random = self.data[h][m]['accuracy_boot_perm'] 
                
                
                #extract those  values for alpha and those for beta
                #take the probability of first class
                data_exactly_list = [arr.item() for arr in data_exactly]
                labels_exactly_list = ['true']*len(data_exactly_list)
                data_random_list= [arr.item() for arr in  data_random]
                labels_random_list = ['random']*len(data_random_list)
                
                #concatenate the two list
                combined_list_data = data_exactly_list + data_random_list
                combined_list_labels = labels_exactly_list + labels_random_list
                
                # Create a DataFrame
                df = pd.DataFrame({'Category': combined_list_labels, 'Value': combined_list_data} )
                sns.histplot(data = df ,x='Value', hue='Category', bins=50, kde=True, palette={'true': 'red', 'random': 'blue'}, multiple="stack",ax =axs[i,j])
                #sns.histplot(x=first_elements_list, bins=100, kde=True, color = 'green',ax =axs[i,j])
                # Customize the legend
                legend_elements = [
                    Line2D([0], [0], color='red', lw=2, label='true'),
                    Line2D([0], [0], color='blue', lw=2, label='random')
                ]
                axs[i,j].legend(title="",handles = legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
               # plt.title("Grouped Histogram")
                axs[i,j].set_xlabel("Accuracy true vs random" )
                axs[i,j].set_ylabel("Counts")
                j += 1
            i += 1
       fig.tight_layout(pad=5)
       fig.suptitle('Hist_accuracy'+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_Histogram_Accuracy_truevsrandom' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       #plt.show()  
       
    '''
       input: dictionary
       output: dataframe
      ''' 
    def create_table(self,hormones,models,class_to_use):
       class1 = str(self.data[hormones[0]][models[0]]['classes'][0])
       class2 = str(self.data[hormones[0]][models[0]]['classes'][1])
       
       data_dict ={'hormones':[],'models' : [], 'accuracy' :[], 'se_accuracy' : [],
                  'precision_' + class1:[], 'se_precision_'+ class1 : [], 'precision_' + class2:[], 'se_precision_' + class2 : [],
                   'recall_' + class1:[], 'se_recall_' + class1:[],'recall_' + class2:[], 'se_recall_' + class2:[],
                   'f_score_' + class1 : [], 'se_f_score_' + class1 :[], 'f_score_' + class2 : [], 'se_f_score_' + class2 :[],
                   '% true values' +  class1 : [], 
                   '% true values' +  class2 : [],
                   'mse' : [], 'r_square' : []}
       
       for  j,h in enumerate(hormones ):
           
            for i,m in enumerate(models):
                data_dict['hormones'].append(h)
                data_dict['models'].append(m)
                list_precision = self.data[h][m]['accuracy']   
                array_mean =100*np.mean(list_precision) 
                array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                data_dict['accuracy'].append(array_mean)
                data_dict['se_accuracy'].append(array_se)
                
                #
                list_precision = self.data[h][m]['precision']   
                array_mean =100*np.mean(list_precision, axis =0) 
                array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                data_dict['precision_' + class1].append(array_mean[0])
                data_dict['se_precision_'+ class1].append(array_se[0])
                data_dict['precision_'+ class2].append(array_mean[1])
                data_dict['se_precision_'+ class2].append(array_se[1])
                #
                list_precision = self.data[h][m]['recall']   
                array_mean =100*np.mean(list_precision, axis =0) 
                array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                data_dict['recall_' + class1].append(array_mean[0])
                data_dict['se_recall_' + class1].append(array_se[0])
                data_dict['recall_' + class2].append(array_mean[1])
                data_dict['se_recall_' + class2].append(array_se[1])
                #
                list_precision = self.data[h][m]['fscore']   
                array_mean =100*np.mean(list_precision, axis =0) 
                array_se = 100*np.std(list_precision, axis=0, ddof=1)/np.sqrt(len(list_precision))
                data_dict['f_score_' + class1].append(array_mean[0])
                data_dict['se_f_score_' + class1].append(array_se[0])
                data_dict['f_score_' + class2].append(array_mean[1])
                data_dict['se_f_score_' + class2].append(array_se[1])
                #
                #
                list_confusion_matrix = self.data[h][m]['confusion_matrix']
                sum_cm = np.sum(list_confusion_matrix, axis=0)
                sum_rows = sum_cm.sum(axis=1, keepdims=True) 
                fraction = (sum_cm/sum_rows)*100
                data_dict['% true values' +  class2].append(fraction[1,1])
                data_dict['% true values' + class1].append(fraction[0,0])
                #
                list_precision = self.data[h][m]['mean_square_error']   
                array_mean =100*np.mean(list_precision) 
                data_dict['mse'].append(array_mean)
                 #
                list_precision = self.data[h][m]['r_square']   
                array_mean =100*np.mean(list_precision) 
                data_dict['r_square'].append(array_mean)
                
        #  Convert into dictionary
        
       total_data = pd.DataFrame(data_dict)
       return   total_data,class1, class2
    
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

        
        
    def __call__(self,select_column_prob):
        keys = plot_data.separate_keys(self.data)
        hormones = list(set(keys[0]))
        models = list(set(keys[1]))
        parameters = list(set(keys[2])) #use set to get unique values
        total_data, class1, class2 = self.create_table(hormones,models,select_column_prob)
        #filter data according to confusion matrix
        total_data_filter_confusion = self.filter_confusion(total_data, class1,class2)
        total_data_filter_all = self.filter_precision(total_data_filter_confusion, class1,class2)
        #create dictionary for each hormone with the correspond model data
        dict_hormones_models = self.create_dict(total_data_filter_all)
        #plot relevant data
       
        if not total_data_filter_all.empty:
          self.plot_confusion_matrix(dict_hormones_models)
          self.plot_precision(dict_hormones_models)
          self.plot_recall(dict_hormones_models)
          self.plot_f1score(dict_hormones_models)
          self.plot_accuracy(dict_hormones_models)
          self.plot_prob(dict_hormones_models,select_column_prob)
      #  self.plot_boot_histograms(hormones,models)


        return total_data, total_data_filter_confusion,total_data_filter_all
     