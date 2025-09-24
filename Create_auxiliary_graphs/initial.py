from rearrange_data import rearrange_data
from plot_data import plot_data

def main():
 #setting 
 path_file = r"F:\SilviaData\rutiFrishman\September2025\Data_hormones.xlsx" 
 obj = rearrange_data(path_file)
 total_data, total_data_selection_hierarchy =obj()
 data_name = total_data.iloc[:,9:40].columns.tolist()
 hormones = total_data.iloc[:,9:13].columns.tolist()
 hormones_ratio = total_data.iloc[:,13:16].columns.tolist()
 end_cann = total_data.iloc[:,16:21].columns.tolist()
 end_cann_ratio = total_data.iloc[:,21:27].columns.tolist()
 aminoacids = total_data.iloc[:,[27,28,29,30,31,36,37,38,39]].columns.tolist()
 aminoacids_ratio = total_data.iloc[:,[32,33,34,35]].columns.tolist()
 #remove no useful columns
 total_data.drop(["Experiment", "Type", "Genotype", "Mice.chips", "Last.day.Glicko", "Animal", "Hierarchy","Days"], axis=1, inplace =True)
 
 obj_plot =  plot_data(total_data)
 obj_plot(hormones, end_cann, 
                       aminoacids,hormones_ratio, end_cann_ratio, aminoacids_ratio)
    
    
    
if __name__ == "__main__":
    main()