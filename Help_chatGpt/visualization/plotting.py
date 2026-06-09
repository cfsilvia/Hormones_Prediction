import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PersonalityVisualizer:
     def plot_archetype_counts(self, labels):
           counts = pd.Series(labels).value_counts()
           plt.figure(figsize=(6,5))
           plt.bar(counts.index.astype(str), counts.values)
           plt.xlabel("Archetype")

           plt.ylabel("Number of mice")

           plt.title("Archetype counts")

           #plt.close()
           return plt.gcf()

     def plot_3d_space(self, X_pca, labels, centers):
           fig = plt.figure(figsize=(10,8))
           ax = fig.add_subplot(111, projection='3d')
           ax.scatter(
            X_pca[:,0],
            X_pca[:,1],
            X_pca[:,2],
            c=labels,
            s=70
        )

           ax.scatter(
                centers[:,0],
                centers[:,1],
                centers[:,2],
                marker="X",
                s=300
            )

           ax.set_xlabel("PC1")
           ax.set_ylabel("PC2")
           ax.set_zlabel("PC3")

           plt.title( "3-Archetype Personality Space")

           #plt.close()

           return fig

     def plot_2d_space(self, X_pca, labels, centers):
           fig = plt.figure(figsize=(7, 6))
           ax = fig.add_subplot(111)
           unique_labels = np.unique(labels)
           cmap = plt.cm.get_cmap("tab10", len(unique_labels))

           for i, label in enumerate(unique_labels):
                 idx = labels == label
                 ax.scatter(X_pca[idx, 0], X_pca[idx, 1],color=cmap(i),s=40,alpha=0.8,label=f"A{label}")
           return fig

