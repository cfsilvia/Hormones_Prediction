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
def create_table(data):
    # Create a dictionary where values are empty lists by default
    data_dict = defaultdict(list)
  
    for m, inner_dict in data.items():
                
                 class1 = str(inner_dict['classes'][0])
                 class2 = str(inner_dict['classes'][1])
                 class3 = str(inner_dict['classes'][2])
                 data_dict['class 1'].append(str(inner_dict['classes'][0]))
                 data_dict['class 2'].append(str(inner_dict['classes'][1]))
                 data_dict['class 3'].append(str(inner_dict['classes'][2]))
                 data_dict['hormones'].append(tuple(inner_dict['features']))
                 data_dict['models'].append(m)                  
                 data_dict['confusion_matrix'].append(inner_dict['confusion_matrix'])
                 data_dict['fscore_' + 'class 1'].append((100*(inner_dict['fscore'] ))[0])
                 data_dict['fscore_' + 'class 2'].append((100*(inner_dict['fscore'] ))[1])
                 data_dict['fscore_' + 'class 3'].append((100*(inner_dict['fscore'] ))[2])
               
                 data_dict['accuracy'].append(100*(inner_dict['accuracy'] ))
                 data_dict['balanced_accuracy'].append(100*(inner_dict['balanced_accuracy'] ))
                
                 data_dict['precision_' + 'class 1'].append((100*(inner_dict['precision']))[0])
                 data_dict['precision_' + 'class 2'].append((100*(inner_dict['precision'] ))[1])
                 data_dict['precision_' + 'class 3'].append((100*(inner_dict['precision'] ))[2])
                 
                 data_dict['recall_' + 'class 1'].append((100*(inner_dict['recall'] ))[0])
                 data_dict['recall_' +'class 2'].append((100*(inner_dict['recall'] ))[1])
                 data_dict['recall_' +'class 3'].append((100*(inner_dict['recall'] ))[2])
                 
                
                # data_dict['se_fscore_' + class1].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[0])
                 
                
                 #data_dict['se_fscore_' + class2].append((100*np.std(inner_dict['fscore'], axis=0, ddof=1)/np.sqrt(len(inner_dict['fscore'])))[1])
                 
                 #confusion matrix
                 sum_cm = np.sum(inner_dict['confusion_matrix'], axis=1)
                 
                 data_dict['% true values' +  'class 1'].append(inner_dict['confusion_matrix'][0,0]*100/sum_cm[0])
                 data_dict['% true values' + 'class 2'].append(inner_dict['confusion_matrix'][1,1]*100/sum_cm[1])
                 data_dict['% true values' + 'class 3'].append(inner_dict['confusion_matrix'][2,2]*100/sum_cm[2])
                 
                 
                 
                         
        #  Convert into dataframe
        
    total_data = pd.DataFrame(data_dict)
    return   total_data,data_dict
    
           
'''
Save pickle data as excel
'''
def save_as_excel(ouput_directory,title_file):
    #read pickle
    with open(ouput_directory + title_file + '.pkl', "rb") as f:
            data = pickle.load(f)
    #Extract the data into a dictionary /save in data frame
    total_data,data_dict = create_table(data)
    mode = 'a' if os.path.exists(ouput_directory +  title_file  + '.xlsx') else 'w'
    with pd.ExcelWriter(ouput_directory + title_file + '.xlsx',mode=mode) as writer:
                 total_data.to_excel(writer, index=False)