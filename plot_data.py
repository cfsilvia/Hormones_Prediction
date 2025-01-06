import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
matplotlib.use('TkAgg') 
plt.ion() # Turn on the interactive mode


class plot_data:
    def __init__(self,data,title,output_directory):
        self.data = data
        self.title = title
        self.output_directory = output_directory
    '''
    input:
    output: prob histogram
    ''' 
    def plot_prob(self, hormones,models,select_column_prob):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
      
       i = 0
       for h in hormones:
            j = 0
            for m in models:
                list_prob = self.data[h][m]['prob'] 
                first_elements = [array[:,select_column_prob] for array in list_prob]#for first class
                list_labels = self.data[h][m]['labels_pred'] 
                class_names = self.data[h][m]['classes'] 
                #extract those  values for alpha and those for beta
                #take the probability of first class
                first_elements_list = plot_data.extract_first_class(first_elements)
                first_elements_labels= plot_data.extract_first_class(list_labels)
                
                # Create a DataFrame
                df = pd.DataFrame({'Category': first_elements_labels, 'Value': first_elements_list} )
                sns.histplot(data = df ,x='Value', hue='Category', bins=50, kde=True, palette={class_names[0]: 'red', class_names[1]: 'blue'}, multiple="stack",ax =axs[i,j])
                #sns.histplot(x=first_elements_list, bins=100, kde=True, color = 'green',ax =axs[i,j])
                # Customize the legend
                legend_elements = [
                    Line2D([0], [0], color='red', lw=2, label=class_names[0]),
                    Line2D([0], [0], color='blue', lw=2, label=class_names[1])
                ]
                axs[i,j].legend(title="",handles = legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
               # plt.title("Grouped Histogram")
                axs[i,j].set_xlabel("Prob." + class_names[select_column_prob])
                axs[i,j].set_ylabel("Counts")
                j += 1
            i += 1
       fig.tight_layout(pad=8)
       fig.suptitle('Histogram'+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_Histogram_Probability' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       plt.show()
            
                

                
    @staticmethod   
    def extract_first_class(first_elements):
        flattened_array = np.concatenate(first_elements)
        first_elements_list = flattened_array.tolist() 
        return  first_elements_list 
          
                
    '''
    input:
    output: precision bar plot
    '''
    def plot_precision(self, hormones,models):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
       i = 0
       for h in hormones:
            j = 0
            for m in models:
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
                axs[i,j].set_ylim(0, 100)
                # Ensure category names are visible
               # axs[i, j].xticks(ticks=range(len(class_names)), labels=class_names, fontsize=12)
                j += 1
            i += 1
            fig.tight_layout(pad=8)
            fig.suptitle('Precision: TP/(TP+FP)'+ '  ' + self.title, x=0.5, y=0.99) 
            plt.savefig(self.output_directory + self.title + '_precision' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
            plt.show()
            
    '''
    input:
    output: accuracy bar plot
    '''
    def plot_accuracy(self, hormones,models):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
       i = 0
       for h in hormones:
            j = 0
            for m in models:
                list_precision = self.data[h][m]['accuracy']   
                array_mean =100*np.mean(list_precision) 
                array_mean_list = array_mean.tolist()
                #get classes
                class_names = self.data[h][m]['classes']
                #bar plot
                axs[i,j].bar("alpha_submissive", array_mean.tolist(), color=['skyblue'], edgecolor = 'black',width = 0.01)
                axs[i,j].set_title(h + '\n' + m, fontsize = 7, color='red')
                axs[i, j].set_ylabel("Accuracy % ", fontsize = 8) 
                axs[i,j].set_ylim(0, 100)
                # Ensure category names are visible
               # axs[i, j].xticks(ticks=range(len(class_names)), labels=class_names, fontsize=12)
                j += 1
            i += 1
            fig.tight_layout(pad=4)
            fig.suptitle('Accuracy: '+ '  ' + self.title, x=0.5, y=0.99) 
            plt.savefig(self.output_directory + self.title + '_accuracy' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
            plt.show()
            
    '''
    input:
    output: recagg bar plot
    '''
    def plot_recall(self, hormones,models):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
       i = 0
       for h in hormones:
            j = 0
            for m in models:
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
                axs[i,j].set_ylim(0, 100)
                # Ensure category names are visible
               # axs[i, j].xticks(ticks=range(len(class_names)), labels=class_names, fontsize=12)
                j += 1
            i += 1
            fig.tight_layout(pad=8)
            fig.suptitle('Recall: TP/(TP+FN)'+ '  ' + self.title, x=0.5, y=0.99) 
            plt.savefig(self.output_directory + self.title + '_recall' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
            plt.show()
    '''
    input:
    output: plot confusion matrix
    '''
    def plot_confusion_matrix(self, hormones, models):
        fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 70)) #good 60
        #fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 15))
        i = 0
        
        for h in hormones:
            j = 0
            for m in models:
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
                j += 1
            i += 1

        fig.tight_layout(pad=8)
        
        fig.suptitle('Confusion matrix'+ '  ' + self.title, x=0.5, y=0.99)
        plt.savefig(self.output_directory + self.title + '.pdf', format='pdf',dpi=300, bbox_inches='tight')
        plt.show()
        
        
    '''
     input:
    output: precision bar plot
    '''
    def plot_f1score(self, hormones,models):
       fig, axs = plt.subplots(len(hormones), len(models), figsize=(20, 60)) 
       i = 0
       for h in hormones:
            j = 0
            for m in models:
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
                axs[i,j].set_ylim(0, 100)
                # Ensure category names are visible
               # axs[i, j].xticks(ticks=range(len(class_names)), labels=class_names, fontsize=12)
                j += 1
            i += 1
            fig.tight_layout(pad=8)
            fig.suptitle('f-score: 2X(precxrecall)/(prec+recall)'+ '  ' + self.title, x=0.5, y=0.99) 
            plt.savefig(self.output_directory + self.title + '_fscore' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
            plt.show()
                
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
       fig.tight_layout(pad=8)
       fig.suptitle('Hist_accuracy'+ '  ' + self.title, x=0.5, y=0.99) 
       plt.savefig(self.output_directory + self.title + '_Histogram_Accuracy_truevsrandom' +'.pdf', format='pdf',dpi=300, bbox_inches='tight')
             # Show the plot
       plt.show()   
        
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
        self.plot_confusion_matrix(hormones, models)
        self.plot_precision(hormones,models)
        self.plot_recall(hormones,models)
        self.plot_f1score(hormones,models)
        self.plot_accuracy(hormones,models)
        self.plot_prob(hormones,models,select_column_prob)
      #  self.plot_boot_histograms(hormones,models)


        a=1
     