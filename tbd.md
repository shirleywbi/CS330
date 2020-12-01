# TODO: Name this file

TODO: This is unsupervised learning

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

## Time series data (L17)

## Survival Analysis (L18)

## Clustering (L19)
