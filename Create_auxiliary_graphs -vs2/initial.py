from rearrange_data import rearrange_data
from plot_data import plot_data
from get_data_interaction import get_data_interaction
from plot_data_hierarchy import plot_data_hierarchy

def main():
 #setting 
 path_file = r"F:\SilviaData\rutiFrishman\September2025\October_2025\data_to_use_complete_without_final.xlsx" 
 obj = rearrange_data(path_file)
 total_data =obj()
 data_name = total_data.iloc[:,8:40].columns.tolist()
 hormones = total_data.iloc[:,8:12].columns.tolist()
 hormones_ratio = total_data.iloc[:,12:15].columns.tolist()
 end_cann = total_data.iloc[:,15:20].columns.tolist()
 end_cann_ratio = total_data.iloc[:,20:26].columns.tolist()
 aminoacids = total_data.iloc[:,[26,27,28,29,30,35,36,37,38]].columns.tolist()
 aminoacids_ratio = total_data.iloc[:,[31,32,33,34]].columns.tolist()
 data_hierarchy = total_data.copy()
 #select alpha , beta and epsilon
 data_hierarchy = data_hierarchy.loc[data_hierarchy['Hierarchy'].isin(["alpha","beta","epsilon"])]
 #change names from dominant to submissive
 
 #remove no useful columns
 total_data.drop(["Experiment", "Type", "Genotype","Hierarchy", "Mice.chips", "Last.day.Glicko", "Animal"], axis=1, inplace =True)
 data_hierarchy.drop(["Experiment", "Type", "Genotype", "Mice.chips", "Last.day.Glicko", "Animal"], axis=1, inplace =True)
 data_hierarchy["Hierarchy"] = data_hierarchy["Hierarchy"].replace({"alpha": "dominant", "beta": "submissive", "epsilon": "submissive"})
 obj_plot =  plot_data(total_data)
 obj_plot(hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
 # select compounds without ratios
 compounds = ['sex'] + hormones + end_cann + aminoacids
 total_data_interaction = total_data.copy()
 total_data_interaction = total_data_interaction[compounds]
 #for clustering   
 obj_interaction_female = get_data_interaction(total_data_interaction[total_data_interaction['sex']=='female'],'female')  
 obj_interaction_female()
 
 obj_interaction_male = get_data_interaction(total_data_interaction[total_data_interaction['sex']=='male'],'male')  
 obj_interaction_male()
 
 #to plotting dominant/submissive
 
#  obj_plot_hierarchy = plot_data_hierarchy(data_hierarchy)
#  obj_plot_hierarchy(hormones, end_cann, 
#                        aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
    
if __name__ == "__main__":
    main()