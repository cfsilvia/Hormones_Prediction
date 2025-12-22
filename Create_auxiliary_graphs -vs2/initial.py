from rearrange_data import rearrange_data
from plot_data import plot_data
from get_data_interaction import get_data_interaction
from plot_data_hierarchy import plot_data_hierarchy
from plot_glycko_correlation import plot_glycko_correlation
from plot_pca import plot_pca
from plot_data_abs_F_M import plot_data_abs_F_M

def main():
 #setting 
 path_file = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\data_to_use\data_to_use_complete_without_final.xlsx" 
 output_dir = r"U:\Users\Silvia\RutiFrishman_2025_hormones_paper\Graphs\Figure2\\"
 choice = 1
 value_plot = 3

 match choice:
    case 1:
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
        #data_hierarchy_glycko = data_hierarchy.copy() #take all status
        data_hierarchy = data_hierarchy.loc[data_hierarchy['Hierarchy'].isin(["alpha","beta","epsilon"])]
        data_hierarchy_glycko = data_hierarchy.copy()
        #change names from dominant to submissive
        
        #remove no useful columns
        total_data.drop(["Experiment", "Type", "Genotype","Hierarchy", "Mice.chips", "Last.day.Glicko", "Animal"], axis=1, inplace =True)
        data_hierarchy.drop(["Experiment", "Type", "Genotype", "Mice.chips", "Last.day.Glicko", "Animal"], axis=1, inplace =True)
        data_hierarchy["Hierarchy"] = data_hierarchy["Hierarchy"].replace({"alpha": "dominant", "beta": "submissive", "epsilon": "submissive"})

        
        #for glycko relation
        #remove no useful columns
        data_hierarchy_glycko.drop(["Experiment", "Type", "Genotype","Hierarchy", "Mice.chips", "Animal"], axis=1, inplace =True)
        #data_hierarchy_glycko["Hierarchy"] = data_hierarchy["Hierarchy"].replace({"alpha": "dominant","beta": "submissive", "gamma": "submissive","delta": "submissive","epsilon": "submissive"})
        match value_plot:
           case 1: 
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
                
                
                
                #do correlation between last day glycko and the rest of proteins 
                obj_plot_glycko = plot_glycko_correlation(data_hierarchy_glycko)
                obj_plot_glycko(hormones, end_cann, 
                                    aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
                
           case 3:
                #to plotting dominant/submissive
                
                obj_plot_hierarchy = plot_data_hierarchy(data_hierarchy)
                obj_plot_hierarchy(hormones, end_cann, 
                                    aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)

                
           case 2:#plot relative
             obj_plot_abs =  plot_data_abs_F_M( output_dir,total_data)
             obj_plot_abs(hormones, end_cann, 
                                    aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio) 
                
              
        
    case 2:
     #pca hormones
     obj = rearrange_data(path_file)
     total_data =obj()
     obj_pca = plot_pca(total_data, output_dir)
     obj_pca()
      
    
if __name__ == "__main__":
    main()