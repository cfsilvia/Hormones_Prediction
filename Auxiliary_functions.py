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