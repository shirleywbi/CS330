# Distances
## Euclidean distance
from sklearn.metrics.pairwise import euclidean_distances
two_cities = cities_X.sample(2, random_state=30)
euclidean_distances(two_cities)

## Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
cosine_similarity(X_train_enc)[1,407]


# Finding the nearest neighbour
## Finding the nearest neighbour
dists = euclidean_distances(cities_df[["lat", "lon"]].iloc[:4])
np.fill_diagonal(dists, np.inf) # To ensure it doesn't return itself as the nearest neighbour
np.argmin(dists[0])

## Finding the distances to a query point (0, 0)
dists = euclidean_distances(cities_X, [[0, 0]])
np.argmin(dists)

## Finding nearest neighbour with NearestNeighbours
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=2) # n_neighbours = # of closest neighbours
nn.fit(cities_X)
# where n_distances is the distance, n_inds is the index of nearest (first index is itself if it's in the dataset)
n_distances, n_inds = nn.kneighbors(cities_X.iloc[[0]]) # Note the double brackets


## KNN Example
grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = X_user_item[grill_brush_ind]

X_user_item = csc_matrix((ratings["rating"], (item_ind, user_ind)), shape=(n_items, n_users)).T
X_item_user = X_user_item.T # transpose to get items x users; where item = observation, users = features

### Assume: A similar item is an item that receives similar reviews by the same people.
nn = NearestNeighbors(n_neighbors=6)
nn.fit(X_item_user)
distances, nearby_items = nn.kneighbors(X_item_user[grill_brush_ind])
nearby_items = np.squeeze(nearby_items)[1:] # Drops itself since its in the dataset

### Total reviews
(X_item_user[grill_brush_ind] > 0).sum()

### Reviewers in common
for item in nearby_items:
    print(np.sum(np.squeeze(X_item_user[grill_brush_ind].toarray()) * np.squeeze(X_item_user[item].toarray()) >0))