import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,          # important if you ever turned it on
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
import matplotlib
matplotlib.use('tkagg') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class plot_pca:
    def __init__(self, data, output_dir):
        self.data = data
        self.data_pca = data.drop(columns = ['Experiment', 'sex', 'Type', 'Genotype', 'Hierarchy', 'Mice.chips', 'Last.day.Glicko', 'Animal'])
        self.results_to_plot = pd.DataFrame()
        self.output_dir = output_dir
    
    def get_pca_components(self):
        data_normalized = StandardScaler().fit_transform(self.data_pca.values)
        pca = PCA(n_components = 3)
        comps = pca.fit_transform(data_normalized)
        pc_df = pd.DataFrame(comps, index = self.data.index, columns = ["PC1", "PC2", "PC3"])
        
        # Merge PCs back into a copy of the original df
        out = self.data.copy()
        out.loc[pc_df.index, "PC1"] = pc_df["PC1"]
        out.loc[pc_df.index, "PC2"] = pc_df["PC2"]
        out.loc[pc_df.index, "PC3"] = pc_df["PC3"]

        #variance explain
        var = pca.explained_variance_ratio_ * 100.0

        #----------Loadings
        feature_names = self.data_pca.columns
        loadings = pd.DataFrame(pca.components_.T, index = feature_names, columns =["PC1", "PC2", "PC3"])

        return out, var, loadings

    def plot_graph_pca(self,out,var,loadings):
       #plot females and males in different colors
       groups = out['sex']
       fig = plt.figure(figsize=(10, 10))
       ax = fig.add_subplot(111, projection='3d')

       colors = ["mediumpurple", "mediumaquamarine"]
       labels_sorted = sorted(groups.unique())
       color_dict = {label: colors[i] for i,label in enumerate(labels_sorted)}


        # compute max range
       max_range = max(
        out["PC1"].max() - out["PC1"].min(),
        out["PC2"].max() - out["PC2"].min(),
        out["PC3"].max() - out["PC3"].min()
       ) / 2



       for label in labels_sorted:
          mask = (groups == label)
          idx = mask & (out["Hierarchy"] == "alpha")
          ax.scatter(out.loc[mask, "PC1"], out.loc[mask, "PC2"], out.loc[mask, "PC3"], label = label, color = color_dict[label], 
                     alpha = 0.6, marker = 'o', s=80 , edgecolors='none')
        #   ax.scatter(out.loc[idx, "PC1"], out.loc[idx, "PC2"], out.loc[idx, "PC3"], label = (label + '_dominant'), color = color_dict[label], 
        #              alpha = 1, marker = 'H', s=200 , edgecolors='none')
          ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
          ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
          ax.set_zlabel(f"PC3 ({var[2]:.1f}%)")
       self.add_arrows(ax,loadings, max_range)
       ax.grid(False)
# ---- REMOVE GRAY BACKGROUND PANES ----
       ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # fully transparent
       ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
       ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

       # Also remove the pane borders
       ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
       ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
       ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

       xmid = (out["PC1"].min() + out["PC1"].max()) / 2
       ymid = (out["PC2"].min() + out["PC2"].max()) / 2
       zmid = (out["PC3"].min() + out["PC3"].max()) / 2

      
       max_range_x = (out["PC1"].max() - out["PC1"].min())/2
       max_range_y = (out["PC2"].max() - out["PC2"].min())/2
       max_range_z = (out["PC3"].max() - out["PC3"].min())/2

      # set limits around center
       ax.set_xlim(xmid - max_range_x, xmid + max_range_x)
       ax.set_ylim(ymid - max_range_y, ymid + max_range_y)
       ax.set_zlim(zmid - max_range_z, zmid + max_range_z)

       ax.legend( bbox_to_anchor=(1.05, 1), loc='upper left')
       ax.set_title("Molecules PCA", fontsize=14, fontweight='bold')

       ax.view_init(elev=20, azim=150)

       plt.tight_layout()
       plt.show()
       #plt.savefig((self.output_dir +"Molecules_pca_plot_3d_vs1.pdf"), format='pdf', bbox_inches='tight')



    '''
     add arrows with the loading
    '''
    def add_arrows(self, ax, loadings, max_range):
        L = loadings[["PC1", "PC2", "PC3"]].values
        norms = np.linalg.norm(L, axis =1) #computes the euclidean distance of each row from the zero
        max_norm = norms.max() if norms.max() != 0 else 1.0
        scale = max_range*0.8 / max_norm #keep arrows inside the plot

        for idx, (feature_name, row) in enumerate(loadings.iterrows()):
         # if norms[idx] > 0.3: 
           x,y,z =row["PC1"], row["PC2"], row["PC3"]
           x,y,z = x*scale, y*scale, z*scale

           #arrow from origin
           ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio = 0.1, linewidth = 1.0, color ="black", linestyle = "--")
           #label near the arrow
           ax.text(x * 1.05, y* 1.05, z*1.05, feature_name, fontsize = 7)

#_____________________________
    def __call__(self):
      out, var, loadings = self.get_pca_components()
      self. plot_graph_pca(out, var, loadings)