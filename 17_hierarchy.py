import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data

# dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Apply hierarchical clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=3)
labels = hierarchical_clustering.fit_predict(X)

# Create a DataFrame for visualization
df = pd.DataFrame(X, columns=data.feature_names)
df['Cluster'] = labels
# Visualize the clusters
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

