# TODO: Name this file

## Distances and Neighbours (L14)

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
