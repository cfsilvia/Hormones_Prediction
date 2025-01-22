import yaml

# Data to be written to YAML
data = {
    'choice': "1",
    '1': {
        'pareto_file': 'F:/Ruti/Pareto/10.09.24_stage2_stage2_Pareto_Information.xlsx',
        'data_file': 'F:/Ruti/PCA/Copy of PCA_PCA__10.9.24_stage2_NEW_stage2_ECBs.xlsx',
        'output_file': 'F:/Ruti/AnalysisWithPython/data_to_use.xlsx',
        'distance' : 0.5
    },
    '2':{ 
        'type': "hierarchy_all_ratios",
        'data_file' : 'F:\Ruti\AnalysisWithPython\data_to_use_complete.xlsx',
        'sex' : 'male',
        'data_features' : 'F:/Ruti/AnalysisWithPython/Better_features_male_hierarchy_all_ratios_400.pkl',
        'n_repeats' : 400,
        'output_directory': "F:/Ruti/AnalysisWithPython/"
        },
    '3': {
        'type': "I_status",
        'data_file' : 'F:\Ruti\AnalysisWithPython\data_to_use_complete.xlsx',
        'sex' : 'male',
        'data_features' : 'F:/Ruti/AnalysisWithPython/Better_features_male_hierarchy_all_ratios_400.pkl',
        'n_repeats' : 400,
        'output_directory': "F:/Ruti/AnalysisWithPython/"
    },
    '4':{
        'type' : "hierarchy_all_ratios_hormones",
        'sex' : 'male',
        'n_repeats' : 400,
        'output_directory' : "F:/Ruti/AnalysisWithPython/",
        
    },
    '5':{
        'type' : "hierarchy_all_ratios",
        'sex' : "female",
        'n_repeats' : 400,
        'data_file' : 'F:/Ruti/AnalysisWithPython/data_to_use_complete.xlsx',
        'choice_m' : "2"
    },
    '6':{
        'type' : "hierarchy_all_ratios_hormones",
        'train_file' : 'F:/Ruti/AnalysisWithPython/data_all_hormones_vs2.xlsx',
        'validation_file' : 'F:/Ruti/fresh_Data/Hormones_Mice_ctrl_casp3_2F_1M_vs1.xlsx',
        'sex' : 'male' ,#female or all
        'choice' : "2",
        'n_repeats' : 1
    },
    '7':{
        #break
    }
}

# Writing data to a YAML file
with open('settings_for_use.yml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print("YAML file 'settings_for_use.yaml' created successfully!")