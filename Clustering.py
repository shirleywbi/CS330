from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

# Comparing Classification vs. Clustering

## Wine dataset
wine = datasets.load_wine()
data_wine = pd.DataFrame(wine.data, columns=wine.feature_names) 
data_wine['target'] = wine.target_names[wine.target]

## Iris dataset
iris = datasets.load_iris() # loading the iris dataset
features = iris.data # get the input data
labels = iris.target_names[iris.target] # get the responses, in this case the specie of the flowers

# Reducing the dimensionality for plotting purposes 
pca = PCA(n_components=2) 
pca.fit(features)
data_iris = pd.DataFrame(pca.transform(features), columns=['$Z_1$', '$Z_2$'])
data_iris['target'] = labels

plot_sup_x_unsup(data_iris, 8, 8)



# k-means clustering
iris_2d = data_iris.iloc[:,:2]

kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(iris_2d)
clusters = kmeans.predict(iris_2d)
plt.scatter(iris_2d.iloc[:,0], iris_2d.iloc[:,1], c=clusters)

kmeans.predict(iris_2d)

kmeans.cluster_centers_ # to get cluster centers

## Cluster examples
housing_raw_clust.query("cluster == 0")[["YearBuilt", "OverallQual", "SalePrice", "GrLivArea", "LotArea"]].mean()
housing_raw_clust.query("cluster == 1")[["YearBuilt", "OverallQual", "SalePrice", "GrLivArea", "LotArea"]].mean()
housing_raw_clust.query("cluster == 2")[["YearBuilt", "OverallQual", "SalePrice", "GrLivArea", "LotArea"]].mean()
housing_raw_clust.plot.scatter(x="YearBuilt", y="SalePrice", 
                               c="cluster", cmap="RdYlBu", alpha=0.5)


## Elbow Method
def plot_elbow(kmeans_dict, elbow=None, w=11, h=5):
    plt.figure(figsize = (w,h))
    if elbow is not None:
        plt.axvline(x=elbow, linestyle='-.', c="black")
    plt.plot(kmeans_dict.keys(), [km.inertia_ for km in kmeans_dict.values()], '-o');
    ax = plt.gca()
    ax.tick_params('both', labelsize=(w+h)/2)
    ax.set_xlabel('K', fontsize=w)
    ax.set_ylabel("Inertia", fontsize=w)

kmeans_sweep_iris = {k : KMeans(n_clusters = k, random_state=1).fit(iris_2d) for k in range(1,10)}
plot_elbow(kmeans_sweep_iris, elbow=3)
kmeans_sweep_iris[3].inertia_

## Silhouette Method
plot_silhouette_dist(16, 8)

## DBSCAN
dbscan = DBSCAN(eps=0.3) # eps is density (distance)
dbscan.fit(X)
plot_clust(X,z=dbscan.labels_)
plt.title("clustering with DBSCAN")



# Hierarchical clustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def plot_dendrogram(w,h, method, metric):

    Z = linkage(votes, method=method, metric=metric)
    w = 22
    h = 10
    fig, ax = plt.subplots(figsize=(w,h))
    dendrogram(Z, labels = votes.index, ax=ax);
    ax = plt.gca()
    ax.set_ylabel("Distance", fontsize=w)
    #ax.set_yticklabels(ax.get_yticklabels(), fontsize=w);
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80, fontsize=w)

plot_dendrogram(22, 10, 'single', 'hamming')

## Linkage
linkage(votes, metric='cityblock')

Z = linkage(votes)
fig, ax = plt.subplots(figsize=(18,10))
dendrogram(Z, 6, truncate_mode='lastp', ax=ax)

fcluster(Z, 6, criterion='maxclust')