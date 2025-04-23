import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


'''
input :shap values 
output order values with male and female 
 '''
def get_Shap(data):
    #get the values
    all_shap_values =  np.concatenate(data['shap_values'], axis = 0) 
    all_features = data['data_features'] 
    feature_names = all_features.columns
    
    #sort the features in increase value according to absolute value
    mean_abs_shap = np.abs(all_shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]
    
    all_shap_values_order = all_shap_values[:,top_indices]
    all_features_order = all_features.iloc[:,top_indices]

    #separate into males and females
    all_shap_values_order_males = all_shap_values_order[getSexIndexes(data['values_sex_test'],'male'),:] 
    all_shap_values_order_females = all_shap_values_order[getSexIndexes(data['values_sex_test'],'female'),:] 
    all_features_order_males = all_features_order.iloc[getSexIndexes(data['values_sex_test'],'male'),:] 
    all_features_order_females = all_features_order.iloc[getSexIndexes(data['values_sex_test'],'female'),:] 
    
    sorted_feature_names = [feature_names[i] for i in top_indices]
    
    return sorted_feature_names,all_shap_values_order_males,all_features_order_males,all_shap_values_order_females,all_features_order_females
    
    
'''
input: indexes where female and males
output: get the respective indexes
'''
def getSexIndexes(data_sex,sex):
    sex_values = [s.iloc[0] for s in data_sex]
    sex_indexes = [i for i,sexs in enumerate(sex_values) if sexs == sex]
    return sex_indexes


'''
input:Draw a gradient-filled violin (without scatter dots) for one group for a given feature,
    then draw its contour using the specified contour_color.

    Parameters:
      ax               : Matplotlib axis object.
      data             : 1D numpy array of SHAP values (group-specific) for one feature.
      feat             : 1D numpy array of feature values corresponding to the SHAP values.
      pos              : y-axis position where the violin is drawn.
      n_bins           : Number of bins along the x-axis (SHAP value axis).
      max_violin_width : Maximum vertical half-width of the violin.
      cmap             : Colormap used to map feature values to colors.
      contour_color    : Color for the contour of the violin.
'''
def draw_gradient_violin(ax, data, feat, pos, n_bins = None, 
                         max_violin_width = None, cmap = None, contour_color = None):
      
   #compute kernel distribution
   kde = gaussian_kde(data)
   x_min, x_max = data.min(), data.max()
   x_range = np.linspace(x_min, x_max, n_bins) #give spaced numbers
   density = kde(x_range)
   density_max = density.max()
   #normalize the features for the colormap
   norm = mpl.colors.Normalize(vmin=np.min(feat), vmax=np.max(feat))
   
   #draw the gradient fill using thin rectangle patches for each x-bin
   for i in range(n_bins - 1):
       x_start = x_range[i]
       x_end = x_range[i+1]
       #density at the middle
       dens_val = float(kde((x_start +x_end)/2))
       half_height = (dens_val/density_max)*max_violin_width
       #points inside the bin
       bin_indices = np.where((data >= x_start) & (data < x_end))[0]
       if len(bin_indices) == 0:
           continue
       #use the median feature to determine the color
       median_feat_val = np.median(feat[bin_indices])
       color = cmap(norm(median_feat_val))
       #draw a rectangle for the x-bin
       rect = patches.Rectangle(
            (x_start, pos - half_height),   # Lower left corner.
            x_end - x_start,                  # Width of the bin.
            2 * half_height,                  # Height (vertical extent).
            color=color,
            linewidth=0,
            alpha=0.8,
            zorder=1
        )
       ax.add_patch(rect)
       
   #create the contour of the violin
   half_heights = (density/density_max)*max_violin_width #for the bins
   top_y = pos + half_heights
   bottom_y = pos - half_heights
   # To create a closed contour, concatenate the top and the reversed bottom.
   contour_x = np.concatenate([x_range, x_range[::-1]])
   contour_y = np.concatenate([top_y, bottom_y[::-1]])
   ax.plot(contour_x, contour_y, color=contour_color, linewidth=2, zorder=4) #zorder means that contour is plot on the top of the other plots
       
 
"""
    Overlay a mini box plot (displaying the interquartile range and median) on the violin.

    Parameters:
      ax         : Matplotlib axis object.
      data       : 1D numpy array of SHAP values for one feature (group-specific).
      pos        : Vertical (y-axis) position to draw the box plot.
      box_height : Thickness of the box (vertical height).
    """
  
def draw_boxplot(ax, data, pos, box_height = None): 

  #comput the 25, median and 75 percentile
  q1,med, q3 = np.percentile(data, [25,50, 75])  
  box_rect = patches.Rectangle((q1, pos-box_height/2),q3-q1, box_height,
                               edgecolor= 'black', facecolor='white', zorder = 2)
  
  ax.add_patch(box_rect)
  # Draw a red vertical line at the median.
  ax.plot([med, med], [pos - box_height / 2, pos + box_height / 2],
            color='red', linewidth=2, zorder=3)
   
   
'''
input: 
output: plot
'''
def plot_violin(sorted_feature_names,all_shap_values_order_males,all_features_order_males,all_shap_values_order_females,all_features_order_females):
    #set up figure
    number_features = 17
    fig, ax = plt.subplots(figsize = (5,number_features*1.2 + 2))
    #Parameters for the violins.
    #cmap = plt.get_cmap("viridis")
    cmap =  LinearSegmentedColormap.from_list(
                'blue_violet_red',
              ['blue', 'violet', 'red']
              )
    n_bins = 5
    max_violin_width = 0.2
    box_height = 0.15
    offset = 0.2 #vertical offset to separate the 2 groups
    used_features = []
    #loop over each feature and draw a violin for female and male
    for index in range(number_features):
        print(index)
        #for male
        male_data = all_shap_values_order_males[:,index]
        male_feat = (all_features_order_males.iloc[:,index]).to_numpy()
        
        pos_male = index - offset
        draw_gradient_violin(ax, male_data, male_feat, pos_male, 
                             n_bins = n_bins, max_violin_width = max_violin_width, 
                             cmap = cmap , contour_color = "blue")
        
       # draw_boxplot(ax,male_data, pos_male, box_height = box_height)
        
        #for female
        female_data = all_shap_values_order_females[:,index]
        female_feat = (all_features_order_females.iloc[:,index]).to_numpy()
        
        pos_female = index + offset
        draw_gradient_violin(ax, female_data, female_feat, pos_female, 
                             n_bins = n_bins, max_violin_width = max_violin_width, 
                             cmap = cmap , contour_color = "red")
        
        # draw_boxplot(ax,female_data, pos_female, box_height = box_height)
        # used_features.append(sorted_feature_names[index])
        
    # Set the y-axis with feature names.
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_yticks(range(number_features))
    ax.set_yticklabels(used_features)
    ax.set_xlabel("SHAP value")
    ax.set_title(" Violin Plots (Contours: Red=Female, Blue=Male)")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
 
'''
 Main function
 '''   
    
def Main_Violin_plots(data):
       sorted_feature_names,all_shap_values_order_males,all_features_order_males,all_shap_values_order_females,all_features_order_females = get_Shap(data)
       plot_violin(sorted_feature_names,all_shap_values_order_males,all_features_order_males,all_shap_values_order_females,all_features_order_females)