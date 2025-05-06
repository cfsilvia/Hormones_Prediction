import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import os


def save_part_of_dict(filename, key, dictionary):
    try:
        # Load existing data if file exists
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        data = {}  # Initialize empty dict if file doesn't exist or is empty
    
    
    # Update the stored dictionary with the new key-value pair
    data[key] = dictionary[key]

    # Save updated dictionary back to the file
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  



'''
Create data frame
'''
def create_table(data,number_labels):
    # Create a dictionary where values are empty lists by default
    data_dict = defaultdict(list)
  
    for m, inner_dict in data.items():
           data_dict['hormones'].append(tuple(inner_dict['features']))
           data_dict['models'].append(m) 
           data_dict['confusion_matrix'].append(inner_dict['confusion_matrix'])
           data_dict['accuracy'].append(100*(inner_dict['accuracy'] ))
           data_dict['balanced_accuracy'].append(100*(inner_dict['balanced_accuracy'] ))
           for index in range(number_labels):   
                 class1 = str(inner_dict['classes'][index])             
                 data_dict['fscore_' + class1].append((100*(inner_dict['fscore'] ))[index])
                 data_dict['precision_' + class1].append((100*(inner_dict['precision']))[index])
                 data_dict['recall_' + class1].append((100*(inner_dict['recall'] ))[index])
            
                 #confusion matrix
                 sum_cm = np.sum(inner_dict['confusion_matrix'], axis=1)
                 
                 data_dict['% true values' +  class1].append(inner_dict['confusion_matrix'][index,index]*100/sum_cm[index])
                         
        #  Convert into dataframe
        
    total_data = pd.DataFrame(data_dict)
    return   total_data,data_dict
'''
Create table for shap values
''' 
def  create_table_shap(data):
    total_data_shap = pd.DataFrame()
    for m, inner_dict in data.items(): #Note:prepare for one model
          
          mice_information = inner_dict['mice_information']
          mice_information= pd.concat(mice_information, ignore_index = True)
          
          classes = inner_dict['classes']
          
          features = (inner_dict['features']).tolist()
          
          shap_values = inner_dict['shap_values']
          shap_values = np.concatenate(shap_values, axis=0)
          shap_values = pd.DataFrame(shap_values)
          shap_values.columns = features
        
          #get abs value and also the order of the features
          shap_abs_value = np.abs(shap_values)
          top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
          
          # order columns of shap values first one with higher feature importance
          shap_values =shap_values.iloc[:,top_indices]
          
          
          labels_pred = inner_dict['labels_pred']
          #labels_pred = pd.DataFrame(labels_pred)
          true_labels = inner_dict['true_labels']
          #true_labels = [(true_labels[i]).iloc[0].item() for i in range(len(true_labels))]
          
          labels_pred_s = [classes[0] if x == 0 else classes[1] for x in labels_pred]
          true_labels_s = [classes[0] if x == 0 else classes[1] for x in true_labels]
          
          total_data_shap = pd.concat([mice_information,shap_values], axis=1)
          
          
          total_data_shap['labels_pred'] = labels_pred_s
          total_data_shap['true_labels'] = true_labels_s
          
          
          
    return total_data_shap
           
'''
Save pickle data as excel
'''
def save_as_excel(ouput_directory,title_file,number_labels):
    #read pickle
    with open(ouput_directory + title_file + '.pkl', "rb") as f:
            data = pickle.load(f)
    #Extract the data into a dictionary /save in data frame
    total_data,data_dict = create_table(data,number_labels)
    mode = 'a' if os.path.exists(ouput_directory +  title_file  + '.xlsx') else 'w'
    with pd.ExcelWriter(ouput_directory + title_file + '.xlsx',mode=mode) as writer:
                 total_data.to_excel(writer, index=False)
                 
    #Save shap values
    shap_values_table = create_table_shap(data)
    mode = 'a' if os.path.exists(ouput_directory +  title_file  +  '_shap_values' +'.xlsx') else 'w'
    with pd.ExcelWriter(ouput_directory + title_file + '_shap_values' + '.xlsx',mode=mode) as writer:
                 shap_values_table.to_excel(writer, index=False)