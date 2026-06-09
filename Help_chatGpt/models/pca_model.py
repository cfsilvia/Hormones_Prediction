from cProfile import label

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.pylab import norm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
import os
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap

mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts




# ============================================================
# PCA
# ============================================================

class PersonalityPCA:

    def __init__(self, n_components=3):

        self.pca = PCA(
            n_components=n_components
        )

    def fit_transform(self, X):

        X_pca = self.pca.fit_transform(X)

        return X_pca

    def explained_variance(self):

        return self.pca.explained_variance_ratio_
    
    def optimize_pca_and_clusters(self, X, pca_range=range(2, 16), cluster_range=range(2, 7), random_state=42):
        results = []
        for n_pcs in pca_range:
             pca = PCA(n_components=n_pcs)
             X_pca = pca.fit_transform(X)
             for k in cluster_range:
                 gmm = GaussianMixture(n_components=k, random_state=random_state)
                 labels = gmm.fit_predict(X_pca)
                 silhouette_avg = silhouette_score(X_pca, labels)
                 results.append({"n_pcs": n_pcs, "n_clusters": k, "silhouette_score": silhouette_avg})

        results_df = pd.DataFrame(results)
        best_row = results_df.loc[results_df["silhouette_score"].idxmax()]
        best_n_pcs = best_row["n_pcs"]
        best_n_clusters = best_row["n_clusters"]

        print(f"Optimal PCA components: {best_n_pcs}, Optimal clusters: {best_n_clusters}, Silhouette Score: {best_row['silhouette_score']:.4f}")
        return results_df
    
    def plot_pca_clusters(self, X, n_pcs, n_clusters, random_state=42, output_dir = None, meta_data=None, glicko = None,  add_hierarchy = False):
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X)
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = gmm.fit_predict(X_pca)
        fig, ax = plt.subplots(figsize=(8, 5))

        # consistent colormap
        # cmap = cm.get_cmap('viridis', n_clusters)
        # colors = cmap(labels )
        cluster_colors = [
    "#440154",  # purple
    "#1b7837",  # green
    "#d95f02" ,  # orange instead of yellow
    "#31688e",  # blue
      ]

        cmap = ListedColormap(cluster_colors[:n_clusters])

        colors = cmap(labels)

        cluster_colors_dominant = [
    "#440154",  # purple
    "#1b7837",  # dark green
    "#d95f02",  # dark orange
    "#2166ac",  # dark blue
]

        cluster_colors_submissive = [ 
    "#9e77c6",  # light purple
    "#80cdc1",  # light green
    "#fdb462",  # light orange
    "#92c5de",  # light blue
]
        

        if glicko is not None:
            # Normalize Glicko for color intensity
            norm_glicko = (glicko - glicko.min()) / (glicko.max() - glicko.min())
            # alpha between 0.2 and 1.0
            colors[:, 3] = 0.3 + 0.7 * norm_glicko

        if add_hierarchy and meta_data is not None:
             colors = np.zeros((len(labels), 4))
             for i in range(len(labels)):
                 cluster = labels[i]
                 if meta_data.iloc[i]["Hierarchy"] == "alpha":
                        colors[i] = to_rgba(cluster_colors_dominant[cluster])
                 else:
                        colors[i] = to_rgba(cluster_colors_submissive[cluster])

        if meta_data is not None:
            marker_map = { "male": "o", "female": "^" }
        
            for sex_value, marker in marker_map.items():
                idx = meta_data["sex"] == sex_value
                ax.scatter(X_pca[idx, 0], X_pca[idx, 1],c=colors[idx], s=70, marker=marker, rasterized=False)
                 # ---------- Marker legend (sex) ----------
                marker_handles = [
                    Line2D(
                        [0], [0],
                        marker=marker,
                        color='w',
                        markerfacecolor='gray',
                        markersize=8,
                        linestyle='None',
                        label=sex_value
                    )
                    for sex_value, marker in marker_map.items()
                ]
                
                   # ---------- Cluster color legend ----------
                cluster_handles = [
                    Patch(
                        color=cmap(i),
                        label=f"Cluster {i}"
                    )
                    for i in range(n_clusters)
                ]

                # Add both legends
                legend1 = ax.legend(
                    handles=marker_handles,
                    title="",
                    loc='upper right'
                )

                ax.add_artist(legend1)

                ax.legend(
                    handles=cluster_handles,
                    title="Archetype / Cluster",
                    loc='upper left'
                )
                        
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"PCA with {n_pcs} components and {n_clusters} clusters")
                
        else: 
                # Cluster legend
            cluster_handles = [
                Patch(
                    color=cmap(i),
                    label=f"Cluster {i}"
                )
                for i in range(n_clusters)
            ]

            ax.legend(
                handles=cluster_handles,
                title="Archetype"
            )

            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA with {n_pcs} components and {n_clusters} clusters")

        if glicko is not None:
            norm = Normalize(vmin=glicko.min(), vmax=glicko.max())

            sm = cm.ScalarMappable(norm=norm, cmap='Greys')
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Last Day Glicko")


        if add_hierarchy and meta_data is not None:
            hierarchy_handles = [
                Line2D(
                    [0], [0],
                    marker='o',
                    color='#4d4d4d',
                    markerfacecolor='#4d4d4d',
                    
                    markersize=8,
                    linestyle='None',
                    label='Dominant (alpha)'
                ),
                Line2D(
                    [0], [0],
                    marker='o',
                    color='#d9d9d9',
                    markerfacecolor='#d9d9d9',
                    
                    markersize=8,
                    linestyle='None',
                    label='Submissive'
                )
            ]

            legend2 = ax.legend(
                handles=hierarchy_handles,
                title="",
                loc='upper left',
                 
            )

            ax.add_artist(legend2)

       # plt.rcParams['pdf.compression'] = 0
        pdf_filename = os.path.join(output_dir, f"PCA_{n_pcs}PCs_{n_clusters}clusters.pdf")
        fig.savefig(pdf_filename, format="pdf", transparent = False, bbox_inches="tight",  dpi=300 )

        plt.tight_layout()

        plt.show()
