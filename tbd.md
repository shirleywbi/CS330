# TODO: Name this file

TODO: This is unsupervised learning

## Nearest Neighbours

`NearestNeighbours` is an example of unsupervised learning, where we try to find the nearest neighbour(s) to a given vector.

- It is advisable to scale your values.

Distance can be defined as euclidean distance, cosine similarity, etc.

### Distance

#### Euclidean distance

The euclidean distance is length of the line segment between two points (Pythagorean distance), as such, it cares about the magnitude of the vectors. This can be applied to vectors in which we get the difference, square and squareroot to get the distance between the vectors (array of numbers, an example).

#### Cosine Similarity

Cosine similarity is the cosine of the angle between two vectors. Unlike euclidean distance, this depends on where the location of the origin "(0,0)" because we draw a line from the origin to the points of interest to compare. It only cares about angle. If we preprocess our features, the origin would be like an "average" case sort of.

- 0 degrees -> cosine of 1 -> most similar
- 180 degrees -> cosine of 0 -> least similar

Because this is a similarity rather than distance, everything is backwards.

- Cosine similarity is between -1 and 1 (similarity < 0 if angle > 90 degrees)
- Cosine distance is between 0 and 2

**Note**: To be consistent with distances, `NearestNeighbours` uses $1 - cosine_similarity$ so smaller values mean more similar.

### Sparse Matrices

Sparse matrices are matrices that only store nonzero elements. Although there is a bit of overhead to store the locations, if the fraction of nonzero is small, it's a win.

- `CountVectorizer` and `OneHotEncoder` output a sparse matrix.
- For a huge number of categories, it may be beneficial to keep them as sparse.
- For a small number of categories, it doesn't matter much.

_**Why don't we create a matrix to find similar items?**_
If we create a matrix of items x users s.t. we use the nearest neighbours to find similar items and the users become the features, the matrix is too big and most of the values will be 0 from missing data.

- **Solution**: Sparse matrices

_**How can I get number of nonzeros?**_
`X_train.nnz`

#### Distances with sparse data

Depending on distance metric, we can get different nearest neighbours.

_**Product Recommendation**_
For the case of product recommendation, it may be better to use cosine similarity because it might be better to recommend more popular items in general. In terms of euclidean distance, because there are a lot of zeros in the dataset, there are arrows in multiple directions. A big more-popular arrow will probably have a larger euclidean distance compared to a small less-popular arrow. In contrast, looking angles doesn't do that.

- Not an issue if scaled but cannot scale because it is a sparse matrix.

## Text Data (L15)

## Outliers

Outliers (anomalies) are observations that are very different from others. Outliers are not categorically good nor bad - it depends on the context (domain specific). Depending on the situation, we might not need to handle them.

One should carefully consider where the outliers might be occurring:

- Train vs. Deploy (`fit` vs. `predict`)
- $X$ vs. $y$ (outliers in features vs. outliers in target)

_**Global vs. Local Outliers**_

- Global outliers: Far from other points
- Local outlier: Within normal data range, but far from other points

_**Why do we want to find outliers?**_

1. Data quality concerns. We want to remove the outliers.
2. Our task is actually to find (identify, detect) the outliers (e.g., credit card fraud)

_**How can we find outliers?**_

- Visually (plots and summary statistics)
- Clustering
- Supervised learning (if labels are available)

_**How can we guard against unwanted outliers?**_  

1. Remove outliers
2. Use methods that are robust to outliers

_**Difficulty with outliers**_

- At what point is it a cluster vs. an outlier? Depends on what you're trying to do.
- It's hard to determine what an outlier is based on text.

_**What should we do with weird deployment data?**_
Use simpler models. Even if you're giving up accuracy, the model is more generalizable and gives a less extreme prediction.

### Methods Robust to Outliers

#### Mean vs. Median

Median is more robust than mean when it comes to outliers because it takes the middle value (and does not get influenced by outliers) compared to means that incorporates the outlier into the sum.

#### Isolation Forests

An isolation forest is a decision tree/random forest that makes totally random splits. If it takes a few splits to isolate an example, then it's more anomalous.

- Requires feature preprocessing

#### RobustScaler

`RobustScaler` is more robust compared to `StandardScaler` and `MinMaxScaler` because the latter two squish the data within a range of either standard deviation or -1 to 1.

#### Quantile Transformer

Instead of looking at the values, `QuantileTransformer` looks at the order. This reduces the effect of outliers but throws away the magnitude. This is less commonly used.

#### Robust Linear Regression

Linear regression is affected by outliers because the coefficients affect all predictions. Linear regression makes predictions and multiplies the coefficient by the feature value. When you have a very large standard deviation above the mean, you would get such a big value for your prediction that it does not want to do it. It prefers to downplay the feature because of the outlier.

Using `HuberRegressor` is more robust to outliers in the **targets**. It even tells you the outliers afterwards.

- `HuberRegressor` behaves about the same as `Ridge` when there are no outliers
- `fit` may be slower

#### Log-transforming Targets

By log transforming targets, it results in less extreme predictions due to smaller values.

#### Random forest

Because random forests split into different paths down the tree, the outlier would only affect the score of the values that fall into that path.

- Random forests are safer than regression.

## Time series data (L17)

## Survival Analysis (L18)

## Clustering (L19)
