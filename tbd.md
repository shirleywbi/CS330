# TODO: Name this file

## Nearest Neighbours

Finding nearest neighbours can be used for both supervised and unsupervised learning.

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

#### Distances with sparse data

Depending on distance metric, we can get different nearest neighbours.

_**Product Recommendation**_
For the case of product recommendation, it may be better to use cosine similarity because it might be better to recommend more popular items in general. In terms of euclidean distance, because there are a lot of zeros in the dataset, there are arrows in multiple directions. A big more-popular arrow will probably have a larger euclidean distance compared to a small less-popular arrow. In contrast, looking angles doesn't do that.

- Not an issue if scaled but cannot scale because it is a sparse matrix.

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

### `NearestNeighbours` in Unsupervised Learning

In unsupervised learning, `NearestNeighbors` It is where we try to find the nearest neighbour(s) to a given vector.

- It is advisable to scale your values.

Distance can be defined as euclidean distance, cosine similarity, etc.

### KNN for Supervised Learning

KNN for supervised learning works by finding the $k$ closest neighbours to a given "query point". This fundamentally relies on a choice of distance. There is KNN for both classifiers `KNeighborsClassifier` and regression `KNeighborsRegressor`.

For regular KNN for supervised learning (not with sparse matrices), you should scale your features.

_**What is k?**_

$k$ is a hyperparameter. A smaller $k$ has lower training error, higher approximation error because it only takes 1 example to change your prediction.

_**Advantages:**_

- Easy to understand, interpret
- Simple hyperparameter ($k$) controlling the fundamental tradeoff
- Can learn very complex functions given enough data

_**Disadvantages:**_

- Can be potentially be VERY slow
- Often not that great test accuracy

#### KNN Regression

In KNN regression, we take the average of the $k$ nearest neighbours.

- Regression plots more natural in 1D, classification in 2D, but we can do either for any $d$

## Clustering

Two ways we can group data is to group the observations such that:

1. Examples in the same group are as similar as possible.
2. Examples in the different groups are as different as possible.

_**What is the difference between classification and clustering?**_
Classification is supervised while clustering is unsupervised (i.e., the labels are unknown). There is no "true" or "correct" cluster, only an optimal cluster. Generally, not even the number of clusters is known.

### $k$-means clustering

1. Assign each example to the closest center.
2. Estimate new centers as average of example in a cluster.
Repeat 1. and 2. until centers and assignments do not change anymore.

- The "classic" k-means is random and can give bad results. Stick to the defaults.

_**`fit`**_

1. Assigns each point to a cluster.
2. Creates a "cluster center" for that point.

_**`predict`**_

- Outputs an array of labels (arbitrary order)

_**k-means vs. KNN**_

- `KMeans` is a clustering algorithm (unsupervised learning) where input is your dataframe, output is assigning each point to a cluster.
- `NearestNeighbors` takes in a point and finds the $k$ closest points.
- `KNeighborsClassifier` and `KNeighborsRegressor` are for supervised learning. They use the nearest neighbours to predict a label.

_**Could there be a deployment phase for clustering?**_
Yes. For example, "what sauce would a new person buy?".

### Choosing $k$

$k$ is the number of clusters. Since in unsupervised learning we do not have the $y$ values, it becomes very difficult to objectively measure the effectiveness of the algorithms. There is no definitive approach (highly subjective).

_**How do we determine how well the clusters "fit" the data?**_
Inertia: Minimal sum of intra-cluster distances

_**Why can't we just look for a $k$ that minimizes the sum of intra-cluster distance (inertia)?**_
Because it decreases as $k$ increases. But, we can evaluate the trade-off: "small k" vs. "small intra-cluster distances".

#### Elbow Method

Find the elbow (visual).
![elbow](https://media.geeksforgeeks.org/wp-content/uploads/20190606105550/distortion1.png)
Ref: Geeks for Geeks

#### Silhouette Method

The silhouette method compares the average distance to the neighbour cluster to the average distance to the given cluster to come up with a silhouette coefficient for that example (sklearn.metrics.silhouette_samples).

- Does not depend on a center

_**Steps:**_
Find the second best cluster for each point.

1. Average the distances of the green point to the other points in the same cluster.
    - These distances are represented by the black lines;
2. Average the distances of the green point to the points in the blue cluster.
    - These distances are represented by the blue lines;
3. Average the distances of the green point to the points in the red cluster.
    - These distances are represented by the red lines

Then, since the average distance to the blue cluster is lower than to the red cluster, the blue cluster is considered to be the neighbour cluster - (the second-best choice for the green point to live, after the black cluster).

_**Silhouette Coefficient:**_
The silhouette coefficients are technically between $-1$ and $+1$ but typically between $0$ and $+1$ (sort of like $R^2$).

- $0$ means close to another cluster, $+1$ means far from other clusters (good)
- This can also be used to compare between different clustering algorithms.

_**Silhouette Coefficient vs. Inertia**_

- Unlike inertia, the silhouette method can be used with any clustering algorithm.
- Unlike the inertia, larger values are better because they indicate that the point is further away from neighbouring clusters.
- Unlike the inertia, the overall silhouette score gets worse as you add more clusters because you end up being closer to neighbouring clusters.
- Thus, as with intertia, you will not see a "peak" value of this metric that indicates the best number of clusters.

##### [Visualizing the silhouette score](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

Horizontal bar chart where each line is one point, sorted from highest to lowest.

- Shows drop-off
  - More rectangular, far to the right is good clustering

### Distance metrics

Distance metrics are used to measure the dissimilarity between points. This is a fundamental step in clustering algorithms because it defines what is similar. But, selection is tricky and there is no methodological recipe for it. It requires domain knowledge.

- `scipy.spatial.distance.cdist` is helpful for calculating distance.
- Be careful when using Euclidean distance on non-scaled data.

_**Do we need to transform our data to numeric?**_
For euclidean distance, yes but not necessarily if our distance function can take in the right data type.

_**Clustering with distances only**_

- Some require the $x$ vectors
  - E.g. $k$-means, for averaging the points to find cluster centres
- Some only require the ability to compute distances between any two points
  - E.g. DBSCAN to verify
- Some only require the distance matrix
  - E.g., `SpectralClustering` with affinity='precomputed'.

### DBSCAN

DBSCAN is a clustering algorithm that does not require you to pick $k$ in advance. It can fit shapes that $k$-means cannot (e.g., inner/outer ring).

- Defines cluster by densitiy. Each point must have at least some number of neighbours in a given radius.
- There is no `predict`. DBSCAN only clusters the points you have, not "new" or "test" points.
- Unlike $k$-means, DBSCAN doesn't have to assign all points to clusters.
  - The label is -1 if a point is unassigned.
- Can only handle one density

_**Needs to:**_

- Define how many neighbour points we require
- The neighborhood size parameter
- The neighborhood shape: This will be defined by the distance metric used.

### Hierarchical clustering

_**How do I handle subclusters of a larger cluster?**_
In a step-by-step way, merge or divide the existing clusters.

_**How to measure clusters' dissimilarity?**_

- Minimum distance (single linkage)
- Maximum distance (complete linkage)
- Average distance (average linkage)

#### Agglomerative clustering

Agglomerative clustering works by merging clusters.

_**Steps**_

1. Start with each point as a separate cluster.
2. Merge the clusters that are most similar to each other.
3. Repeat Step 2 until you obtain only one cluster ($n-1$ times).

##### Understanding the output of `linkage`

The function linkage will return a matrix (n-1)x4:

- First and second columns: indexes of the clusters being merged.
- Third column: the distance between the clusters being merged.
- Fourth column: the number of elements in the newly formed cluster.
The rows represent the iterations.

There are several ways to truncate the tree in the dendrogram (especially when you have a big $n$).

## NLP

Natural Language Processing (NLP) involves extracting information from human language.

Examples:

- Translation
- Summarization
- Sentiment analysis
- Relationship extraction
- Question answering / chatbots

_**Why is NLP difficult?**_

- Lexical ambiguity
  - e.g., Who is Panini? Person vs. Sandwich
- Part-of-speech ambiguity
  - e.g., Fruit flies like a banana.
    - Flies = preposition vs. noun?
    - Like = preposition or verb?
- Referential ambiguity
  - e.g., If the baby does not thrive on raw milk, boil **it**

### Word Counts, TF-IDF

Similar to `CountVectorizer`, but you normalize word count by the frequency of the word in the entire dataset.

- EX. If "earthshattering" appears 10 times, that is more meaningful than if "movie" appears 10 times because it is less common.
- Has the same shape as `CountVectorizer` but the counts are normalized
- `TfidfTransformer` can take the word counts from `CountVectorizer` and transform them to the same output as `TfidfVectorizer`.

### Word embeddings

Word embeddings is "embedding" a word in a vector space. You have a bunch of feature columns and each word has a representation.

- Can result in sparse and dense embeddings

_**Sparse vs. Dense word vectors**_

- Term-term and term-document matrices are sparse
- OK because there are efficient ways to deal with sparse matrices

_**Alternatives**_

- Learn short (~100 to 1000 dimensions) and dense vectors.
- These short dense representations of words are referred to as word embeddings.
- Short vectors may be easier to train with ML models (less weights to train).
- They may generalize better.

_**How can we get dense vectors?**_

- Count-based methods
  - Singular Value Decomposition (SVD) - beyond the scope of the course
- Prediction-based methods
  - Word2Vec - able to capture complex relationships between words
    - EX. What is the word that is similar to WOMAN in the same sense as KING is similar to MAN?
    - Perform a simple algebraic operations with the vector representation of words: $\vec{X} = \vec{KING} - \vec{MAN} + \vec{WOMAN}$
    - Search in the vector space for the word closest to $\vec{X}$ measured by cosine distance.
  - fastText
  - GloVe

_**Implicit biases and stereotypes in word embeddings**_
Word embeddings reflects gender stereotypes present in broader society. They may also amplify these stereotypes because of their widespread usage.

#### Vector space model

In vector space models, we model the meaning of a word by placing it in a vector space. Distances among words in the vector space indicate the relationship between them.

!['vector-space-model'](https://github.com/UBC-CS/cpsc330/raw/8c4bcfe4785d680b5201449a8ce24b7298a6ed08/lectures/img/t-SNE_word_embeddings.png)

(Attribution: Jurafsky and Martin 3rd edition)

#### Distributional hypothesis

The distributional hypothesis suggests "You shall know a word by the company it keeps. If A and B have almost identical environments we say that they are synonyms."

EX.

- Her child loves to play in the playground.
- Her kid loves to play in the playground.

#### Co-occurrence matrices

A way to represent vectors into a vector space.

#### Term-document matrix

A term-document matrix is the transpose of `CountVectorizer` For each word, what are the documents it appears in. Similar words are the words that appear in similar documents.

- Each cell is a count of words in the document in that column.
- You can describe a document in terms of the frequencies of different words in it.
- You can describe a word in terms of its frequency in different documents.

#### Term-term matrix

The idea of a term-term matrix is to go through a corpus of text, keeping a count of all of the words that appear in its context within a window.

#### Pre-trained embeddings

Pre-trained skip the step of training complicated ML models on big data sets by having someone else do the `fit` for us already. We can just download and `transform`.

Examples:

- word2vec
  - trained on several corpora using the word2vec algorithm, published by Google
- GloVe
  - trained using the GloVe algorithm, published by Stanford
- fastText pre-trained embeddings for 294 languages
  - trained using the fastText algorithm, published by Facebook

## Feature Engineering

Feature engineering is the general task of coming up with good features given available input data.

- In the past, this was often done "manually" for things like images, text, etc. But now a lot of this has moved to deep learning, and often pre-trained models.
- You can engineer whatever features you want, e.g. # bathrooms per bedroom
- For a super complex model, this helps with overfitting (prior knowledge)
- For a super simple model, this helps with underfitting (also prior knowledge)

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

## Time series data

### Splitting Temporal Data

#### Train/test splits

Unlike with other data where we can use `train_test_split`, we cannot use it with temporal data because it means that we are training on data that came before our test data. If we want to forecast, we aren't allowed to know what happened in the future. Instead, we can split the data at a given time point.

#### Cross-validation

Cross-validation randomly shuffles the rows, so you might get rows in your training set that occur after rows in your validation set. This isn't 100% terrible in that you're still predicting next week's values but it's not a good idea, especially if there are trends.

- Our training split is a bit weird because we "have the answers" in the training set.

#### Other splitting methods

Some other splitting methods are `TimeSeriesSplit` and `ShuffleSplit`. 

_**`TimeSeriesSplit`**_

- Expects dataframe to be sorted by date
- OK to have multiple measurements for a given timestamp

### Encoding date/time as feature(s)

_**Encoding methods**_

- Encoding time as a number
- OHE of the month
- OHE of the seasons
- Periodic encoding (sin/cos over a year)

### Lag-based features

A lag feature is a variable which contains data from prior time steps. We can also create a lagged version of the target.

- Only works with equally spaced data, otherwise does not make sense

_**Is it fine to add a lag feature?**_
Yes, if you would have this information available in deployment.

_**Is there a difference between adding a lag feature then splitting vs. splitting then adding a lag feature?**_
Yes, a tiny difference. The first day of the test set, for each location, will have `NaN`

- That might actually be too strict
- It should be fine to do this before splitting

### Forecasting further into the future

_**Approaches:**_

1. Train a separate model for each number of days (Recommended).
2. Use a multi-output model that jointly predicts tomorrow, in 2 days, etc.
3. Use one model and sequentially predict using a for loop. However, this requires predicting all features into a model so may not be that useful here.

### Trends

With `LinearRegression`, we can learn a coefficient for `Days_since`. If the coefficient is positive, it predicts unlimited growth forever.

With a random forest, we'll be doing splits form the training set (e.g., "if `Days_since` > 9100 then do this"). There will be no splits for later time points because there is no training data there. Therefore, tree-based models CANNOT model trends.

Often, we model the trend separately and use the random forest to model a de-trended time series.

## [Survival Analysis](https://github.com/UBC-CS/cpsc330/blob/master/lectures/18_survival-analysis.ipynb)

Survival analysis is when we want to analyze the time until an event occurs.

Examples:

- The time until a disease kills its host
- The time until a piece of equipment breaks.
- The time that someone unemployed will take to land a new job.
- The time until a customer leaves a subscription service.

### Censoring

Censoring is an occurrence that prevents you from observing the exact time that the event happened for all units/indidividuals that are being studied. This means that we don't have correct target values to train/test our model.

_**Types:**_

- Right censoring
  - Did everyone join at the same time?
  - Are there other reasons the data might be censored at random times (e.g., death)?
- Left censoring
- Interval censoring

_**Why can't we use regular regression models?**_
For a customer who has churned, they have left so the tenure is correct. However, for customers who have not churned, they have stayed for AT LEAST the tenure. This leads to an underestimate.

#### Approach 1: Consider only the cases for which we have the time

_**Issue:**_
On average they will be underestimates (too small), because we are ignoring the currently subscribed (un-churned) customers. Our dataset is a biased sample of those who churned within the time window of the data collection. Long-time subscribers were more likely to be removed from the dataset! This is a common mistake.

#### Approach 2: Assume everyone churns right now (i.e., use original dataset)

_**Issue:**_
It will be an underestimate again. For those still subscribed, while we did not remove them, we recorded a total tenure shorter than in reality, because they will keep going for some amount of time. because we have a bunch of churns "now" that did not actually happen.

#### Approach 3: Survival analysis

`lifelines` is a package for survival analysis that takes into consideration people who have left and people who have stayed at least this long. It can answer:

- How long do customers stay with the service?
- What factors influence a customer's churn time?
- For a particular customer, can we predict how long they might stay with the service?

### Kaplan-Meier Curve

Kaplan-Meier Curve shows the probability of survival over time (i.e., after certain months, what is the probabilty they're still around).

- Looks only at tenure and churn
- Individual K-M curves can be applied to different subgroups
  - Does not look at features

### Cox Proportional Hazards Model

The Cox proportional hazards model is a commonly used model that allows us to **interpret how features influence a censored tenure/duration**.

- Like linear regression for survival analysis: we will get a coefficient for each feature that tells us how it influences survival
- It makes some strong assumptions (the proportional hazards assumption) that may not be true
- The proportional hazard model works multiplicatively, like linear regression with log-transformed targets

### Prediction

Prediction focuses on indiviudal customers.

With regular supervised learning, tenure was a feature and we could only predict whether or not they had churned by then.
