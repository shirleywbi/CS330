# CPSC 330 Summary Sheet: Supervised Learning on Tabular Data

## Supervised Learning

Given a set of features $X$, predict associated targets $y$.

There are two types of models:

- **Classification** = categorical target
- **Regression** = quantitative target

_**Workflow**_

1. Immediately split into train/test (before any exploratory data analysis).
2. Use cross-validation on the training split for your model validation.
3. Test your model once on the test set at the very end.

## Golden Rule

The test data should not influence the training process in any way. If the test data were to influence the training data, the model will be overly optimistic. The test data is used to act as a proxy for deployment error so it should only be used once.

Still, it may not be a perfect representation due to:

- Bad luck (worse for smaller datasets)
- Deployment data coming from a different distribution
- Test data is used more than once

## Sources of Error

Because the goal of supervised learning is to predict unseen/new data, we do not want the model to be too closely modeled to our training data. As a result, we split the data into **training** and **testing**.

- Data should be split randomly

_**Why won't you ever have $E_{best} = 0$?**_
Because the target is not a fixed deterministic function of the features. There is always some randomness.

### Fundamental Trade-off

As you increase model complexity, $E_{train}$ tends to go down but $E_{test} - E_{train}$ tends to go up.

### Overfitting

Overfitting occurs when the model is too close to the model. Positively correlated with complexity. With an infinite amount of training data, overfitting would not be a problem.

_**Occurs when:**_

- $E_{train} < E_{best} < E_{test}$
- Train score is much better than validation/test score

#### Overfitting the validation set

- Small datasets make it harder to trust the best hyperparameters and accuracy due to noise

_**What should I do if test score << cross-validation score?**_

- Realistically report the values
- Use the test set a couple of times
- Try simpler models

### Underfitting

Underfitting occurs when the model is so simple that it does not pick up the patterns from the training set.

_**Occurs when:**_

- $E_{best} < E_{train} ~< E_{test}$
- Train score is poor, and is similar to validation/test score

## Classifiers

_**What does .score() do?**_
It calls predict then compares the predictions to the true labels.

_**Why would score not result in 100% accuracy?**_
There may be instances of duplicate features with different target values.

### Multi-class classification

Multi-class classification refers to classification with >2 classes.

_**How does multi-class classification differ from binary classification?**_

- Same for feature importance
- `predict_proba` outputs one value per class
- Outputs one coefficient per feature per class in `LogisticRegression`
  - Sums to 0 for each feature
  - By convention
  - Only occurs with `multi_class='multinomial'`

If you want general feature importance irrespective of class, you can look at the sum of the squares of the coefficients.

`(lr_coefs**2).sum(axis=1).sort_values(ascending=False)`

_**How do precision, recall, etc. work?**_
Precision, recall, etc. do not apply directly, but if we pick one of the classes as positive and the rest negative, then we can.

### DummyClassifier

DummyClassifier makes predictions based on the target value, ignoring the features. Any reasonable classifier should have a higher prediction accuracy than `DummyClassifier`.

- May not always get >= 50% prediction accuracy (e.g., when there are more than 2 target values)

### Decision Trees

Decision trees are classifiers that make predictions by sequentially looking at features and checking whether they are above/below a threshold. They learn axis-aligned decision boundaries (vertical and horizontal lines with 2 features).

- Categorical - yes/no questions
- Numeric - thresholds (e.g., 5%, 50%)

_**Why not use a very deep tree?**_
Although it will result in larger accuracy of the training data, it may result in overfitting and not apply to the test/deployment data.

_**Why don't we need to scale?**_
Because decision trees work with thresholds rather than absolute values, we don't need to scale.

#### Random Forests

Random forests are a collection of decision trees. For instance, `RandomForestClassifier` is an average of a bunch of random decision trees. They are ensembles built using bootstrapping.

- Each tree "votes" on the prediction, majority rules
- Each tree (and split) is limited in the number of features it can look at
- Each tree is training on a slightly different version of the dataset

_**Available Hyperparameters**_
These classifiers implement `predict_proba` so we can use something like ROC AUC. The probability scores come from the variation in the votes across trees, and other fancier sources. But, not all of them implement `class_weight`.

##### Gradient Boosting Tree Classifier

Gradient boosting tree-based classifiers are complicated random forests. Some examples are `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`.

### Logistic Regression (Linear Classifier)

Logistic Regression is a popular linear classifier that consists of input features, coefficients (weights) per feature, and bias or intercept.

- Will fail if non-numeric data

_**Advantages:**_

- Popular
- Fast training and testing (e.g., huge datasets)
- Interpretability
  - Coefficients (weights) are how much a given feature changes the prediction and in what direction

_**predict_proba**_
Unlike `predict` that outputs the value with the highest confidence score, `predict_proba` outputs a list of confidence scores on a set of given features.

_**C**_
C is correlated to the complexity of a model. Smaller C leads to less confident predictions (probabilities closer to 0.5).

### Ensembles

Ensembles create multiple models and combine their results. Ensembles are effective but tradeoff code complexity and code speed for prediction accuracy. With hyperparameter optimization, there is added slowness. Two methods of ensembling are stacking and averaging.

#### Averaging

`VotingClassifer` will take a vote using the predictions of the constituent classifier pipelines.

- `voting='hard'` uses the output of predict and actually votes
- `voting='soft'` averages `predict_proba` and thresholds / takes the larger.
- The choice depends on whether you trust `predict_proba` from your base classifiers.

|Example|log reg|rand for|cat boost|Averaged model|
|-|-|-|-|-|
|1|✅|✅|❌|✅✅❌=>✅|
|2|✅|❌|✅|✅❌✅=>✅|
|3|❌|✅|✅|❌✅✅=>✅|

_**Advantages:**_

- Potentially getting a better classifier by averaging multiple together

_**Disadvantages:**_

- `fit`/`predict` time
- Reduction in interpretability
- Reduction in code maintainability

_**Why can averaging the models improve the prediction?**_
The different models make different mistakes.

#### Stacking

Use the output of a model (features) as the inputs to another model. By default, this uses `LogisticRegression`, resulting in a weighted average of otuputs and learning from the weights. An effective strategy is randomly generating a bunch of models with different hyperparameter configurations, and then stacking all the models.

- The number of coefficients = the number of base estimators
- By default, it does cross-validation, fitting the base estimators on the training fold, predicting on the validation fold, and then fitting the meta-estimator on the output (on the validation fold)

_**Advantages:**_

- More accuracy compared to voting
- Can see the coefficients for each base classifier

_**Disadvantages:**_

- Slower than voting

## Regression

Similar to classifiers, there are corresponding estimators in regression:

- DummyClassifier -> DummyRegressor
- LogisticRegression -> Ridge
- RandomForestClassifier -> RandomForestRegressor
- VotingClassifier -> VotingRegressor
- XGBClassifier -> XGBRegressor
- etc.

### DummyRegressor

`DummyRegressor` is a regressor that makes predictions using simple rules. This regressor is useful as a simple baseline to compare with other regressors.

### Linear Regression (Ridge)

Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables. We will use `Ridge()` because there is more flexibility with hyperparameter `alpha`.

- Using Ridge also allows us to use `handle_unknown='ignore'` in OHE.
- `alpha` is the inverse of `C`
  - Smaller `alpha`: lower training error
  - Larger `alpha`: lower approximation error
- Generally larger `alpha` leads to smaller coefficients
  - Smaller coefficients mean the predictions are less sensitive to changes in the data, meaning less overfitting
- Avoid `alpha=0` because `Ridge(alpha=0)` is equivalent to `LinearRegression`
- Coefficient size will not change in the same proportion
- Hyperparameters can be tuned with `RidgeCV`

_**Advantages:**_

- Basic or popular
- Interpretable

### Ensembles (Regression)

Similar to classifiers, there is ensembling in regression:

- VotingClassifier -> VotingRegressor
- StackingClassifier -> StackingRegressor

_**How does ensembling differ in regression?**_

- **Averaging**: Calculates the average between multiple estimators
- **Stacking**: Meta-model is `Ridge`
  - Instead of using `predict_proba` to get numerical inputs, since the data is already numeric, we can just use the output of `predict`.

## Hyperparameter Selection (Cross-validation)

To tune our hyperparameters, we need to add an additional split for validation so that our test data remains unseen (to simulate deployment data). In total, we have 4 datasets: train, validation, test and deployment.

||fit|score|predict|
|-|-|-|-|
|Train      |✔️ |✔️  |✔️|
|Validation |   |✔️  |✔️|
|Test       |   |once|once|
|Deployment |   |    |✔️|

### $k$-fold cross-validation

To obtain a better estimate of test error than just using a single validation set by using multiple splits on all the data. Each fold turns into a validation set and the values for each fold are averaged.

_**Advantage**_

- More accurate score estimates, thus better models
- Gives a measure of uncertainty in the scores

_**Disadvantage**_

- Speed
- More complicated code

_**Finding the optimal model**_
Pick the model with the lowest validation error (reasonable for now) and low approximation error (if possible)

### Manual hyperparameter optimization

_**Advantage:**_

- We may have some intuition about what might work (e.g., if massively overfit, we know to try decreasing `max_depth` or `C`).

_**Disadvantage:**_

- It takes a lot of work.
- In very complicated cases, our intuition might be worse than a data-driven approach.

### Automatic hyperparameter optimization

Two methods that can be used are exhaustive grid search and randomized search.

- Allows for a custom scoring function (e.g., `scoring='roc_auc'`)

_**Advantage:**_

- Reduce human effort
- Less prone to error and improve reproducibility
- Data-driven approaches may be effective

_**Disadvantage:**_

- May be hard to incorporate intuition
- Be careful about overfitting on the validation set

#### Exhaustive grid search (GridSearchCV)

A user specifies a set of values for each hyperparameter. The method considers the "product" of the sets and then evaluates each combination one by one.

_**How many models?**_
$n_Models = |C| * |folds|$

_**Disadvantages:**_

- Required number of models to evaluate grows exponentially with the dimensionality of the configuration space
- May become infeasible fairly quickly

#### Randomized hyperparameter optimization (RandomizedSearchCV)

Optimization samples configurations at random until certain budget (e.g., time) is exhausted.

_**How many models?**_
$n_Models = n_iter * |folds|$

_**Advantages:**_

- You can choose how many runs you'll do.
- You can restrict yourself less on what values you might try.
- Adding parameters that do not influence the performance does not affect efficiency.
- Research shows this is generally a better idea since you don't know in advance which hyperparameters are important for your problem. GridSearchCV is more likely to result in repeatedly choosing unimportant parameters.

## Transformers

Transformers convert the data, often used for data cleaning, transforming text to numbers, etc. This needs to be done before preprocessing.

_**How can I combine complex pipelines?**_
To combine multiple types of transformers, use `ColumnTransformer`, which allows different operations on different sets of columns in parallel.

![column-transformer](https://github.com/UBC-CS/cpsc330/raw/030b01fda146513d90cfc4bc15940ba6897ba345/lectures/img/column-transformer.png)
[Source](https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/#37)

- `ColumnTransformer` fits all the transformers and transforms all the transformers when they are called.
- `ColumnTransformer` throws away any columns not accounted for in its steps, unless `remainder='passthrough'` is set.

To get the hyperparameter names from the pipeline, we can use `pipe.get_params().keys()`.

### CountVectorizer

Transforms text data into counts or present/absent. May need to group values to avoid overfitting.

### Encoding categorical variables

#### No encoding

Dropping the data, but discards a lot of information

#### OrdinalEncoder

Transforms categorical variables by assigning an integer (ordered) to each unique categorical label.

#### OneHotEncoder

Transforms categorical variables into a different column for each value (0 = applies, 1 = does not apply)

_**How do we deal with unknown values?**_

- Add `handle_unknown='ignore'` to avoid errors which sets 0 for unknown values
- If we know the categories, it might be reasonable to "violate the Golden Rule" by looking at the test set and hard-coding the categories

_**Using drop='first'**_
_Pros:_

- In certain cases, like `LinearRegression`, this is important
- Technically redundant to not do this

_Cons:_

- Prevents using `handle_unknown='ignore'` which is very often useful
- Makes feature importances more confusing
- Occasionally the choice of which feature gets dropped matters (e.g., feature selection after OHE)

_**drop='if_binary'**_

- Arbitrarily chooses one of the two categories based on the sorting and drops the first one.
- Does not work with `handle_unknown='ignore'` so you may end up skipping it for convenience.

_**Things to consider:**_

- You may want to pre-set the categories
- You may want an "other" category
- You may want only one column for binary variables
- Avoid `drop='first'`

### Handling missing data

#### Dropping data

One way to handle missing data is to drop the examples with missing data. However, we will not know how to handle missing values in deployment and the missing values may not be random.

`X_train_nan.dropna(axis=1).shape`

#### Imputation (SimpleImputer)

Imputation invents values for the missing data.

For categorical data, use `strategy='most_frequent'` or `strategy='constant', fill_value='?'`.
For numerical data, use `strategy='mean'` or `strategy='median'`.

### Scaling numerical values

Two methods of scaling is standardization and normalization:

|Approach|What it does|How to update $X$|sklearn implementation|
|-|-|-|-|
|normalization|sets range to $[0, 1]$|`X -= np.min(X,axis=0)` <br/>`X /= np.max(X,axis=0)`|`MinMaxScaler()`|
|standardization|sets sample mean to 0, S.D. to 1|`X -= np.mean(X,axis=0)` <br/> `X /= np.std(X,axis=0)`|`StandardScaler()`|
||removes the median and scales according to the quantile range||`RobustScaler`|

_**Why conduct feature scaling?**_

- Improves performance for some models (e.g., LogisticRegression, not decision trees)
- Is generally a good idea for numeric features
  - May get very small coefficient values if values are very big (e.g., in LogisticRegression)

## Pipelines

Sequentially apply a list of transforms and a final estimator, refitting again on each fold.

_**Advantages:**_

- Cleanly call `fit` on train split and `score` on test split
- Can't accidentally re-fit the preprocessor on the test data
- Automatically makes sure the same transformations are applied to train and test

![pipeline](https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/images/pipeline.png)
[Source](https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/images/pipeline.png)

_________________________________

## Analysis

By default, `.score()` outputs accuracy but there are other ways to analyze the model such as:

- Precision
- Recall
- F1 score

_**Why should we look at scores other than accuracy?**_
Accuracy does not tell the whole story! Consider imbalanced datasets which may have a very high accuracy (e.g., 99.9%). That information alone is not enough to tell us whether a model is good because it is so uncommon.

- EX. Fraud, medical results, etc.

_**Which metric do we care about?**_
Although we would like high precision and recall, the balance depends on our domain. For instance, recall is really important if we really need good detection (e.g., credit card fraud detection).

_**How can we interpret the scores?**_

- High std for test score -> We can't exactly trust the ordering
- Higher train scores compared to test scores -> Overfitting
  - Adjust hyperparameters

_**What should we consider when analyzing and making our conclusions?**_

- Size of the dataset
- CV folds
- Test set
- Choice in scoring metric

### Confusion Matrix

Confusion matrices allow us to visualize predicted vs. actual class.

- Perfect prediction has all values down the diagonal.
- Off diagonal entries can often tell us about what is being mis-predicted.

|X|predict negative|predict positive|
|-|-|-|
|negative example|True negative (TN)|False positive (FP)|
|positive example|False negative (FN)|True positive (TP)|

### Evaluation Metrics

#### Accuracy

$Accuracy = \frac{TN + TP}{TN + FN + TP + FP}$

#### Precision

"Of the positive examples you identified, how many were real?"

$Precision = \frac{TP}{TP + FP}$

#### Recall

"How many of the actual positive examples did you identify?"

$Recall = \frac{TP}{TP + FN}$

#### F1 Score

F1 score is a metric that "averages" precision and recall. If both precision and recall go up, the F1 score will go up, so in general we want this to be high. F1 score is for a given threshold and measures the quality of `predict`.

- Can be used if we need a single score to maximize (e.g., `RandomizedSearchCV`)
- Accuracy is often a bad choice

![evaluation-metrics](https://github.com/UBC-CS/cpsc330/raw/030b01fda146513d90cfc4bc15940ba6897ba345/lectures/img/evaluation-metrics.png)
Source: Varada Kolhatkar

#### Precision-Recall Curves

A precision-recall curve computes and plots a grid of possible thresholds

![precision-recall curve](https://scikit-learn.org/stable/_images/sphx_glr_plot_precision_recall_001.png)

- _**Top-right:**_ Perfect classifier, where precision = recall = 1
- _**Red star:**_ Threshold = 0.5
- **Average precision (AP)** is the area under the curve and is a score that summarizes the "goodness" of the plot. It is a summary _**across**_ thresholds and measures the quality of `predict_proba`.
- Bumpy because changing the threshold affects some discrete number of examples

#### ROC Curves

ROC plots true positive (recall) against false positive rate

![roc-curve](https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png)

- _**Diagonal line**_: If random guesses
- _**Top-left**_: Best
- _**Threshold = 1**_: Always predict "negative" (bottom-left)
- _**Threshold = 0**_: Always predict "positive" (upper-right)
- Curve is monotonic

##### Area under the curve (AUC)

AUC provides a single meaningful number for systems performance

- AUC = 1.0 means perfect classification
- AUC = 0.5 means random chance

_**Why is ROC AUC not highly influenced by class_weight?**_
Changing the `class_weight` is like changing the thresholds. Since ROC AUC is a summary of the thresholds, it doesn't change much.

![auc](https://camo.githubusercontent.com/f56a65760302c86e4f0d446136db43c58ca37936/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f646172697961737964796b6f76612f6f70656e5f70726f6a656374732f6d61737465722f524f435f616e696d6174696f6e2f616e696d6174696f6e732f6375746f66662e676966)

Consider the following:

- Let:
  - X-axis = predict proba
  - blue = fraud
  - orange = not fraud
- If the threshold is at a position where everything on its right is blue, there is 100% precision with that positive class. However, there is lower recall. As we move the threshold left, there is a tradeoff between precision and recall. Precision decreases and recall increases.

![auc-2](https://camo.githubusercontent.com/992d8f60f2697455518553d570dbdd7b7c0bdd27/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f646172697961737964796b6f76612f6f70656e5f70726f6a656374732f6d61737465722f524f435f616e696d6174696f6e2f616e696d6174696f6e732f524f432e676966)

Consider the following cases:

1. **No overlap**
    - ROC curve is a right angle
    - Always right because there are no cases where you would mispredict.

2. **Some overlap**
    - ROC curve is between diagonal and right angle
    - As recall increases, so does false positive

3. **Full overlap**
    - ROC curve shows a rough diagonal line.
    - It is a useless classifier. Try a different one

#### Communicating Confidence: Credence and `predict_proba`

**Credence** means truthfulness or believability. Avoid overestimating your confidence in a score. We don't just want to be right. We want to be confident when we're right and hesitant when we're wrong.

In practical terms, it means that:

- I would accept a bet at these odds
  - For a chance of winning $1, I would bet $99 that I'm right about this.
- Long-run frequency correctness
  - 75% sure means that for every 100 predictions I make at this level of confidence, I would expect about 25% to be incorrect.

This is along the lines of `predict` vs. `predict_proba`.

_**What does it mean to be 0%, 50%, 100% sure?**_

- 0% -> 100% sure it's not the case
- 50% -> Unsure
- 100% -> 100% sure it is the case

##### Loss Functions

Loss functions are used to quantify confidence in `predict_proba`.
When calling `fit` for `LogisticRegression` it has the same preferences: correct and confident > correct and hesitant > incorrect and hesitant > incorrect and confident

### Class Imbalance

Training sets are imbalanced when there are more examples in one class than another (e.g., 1:10 ratio)

_**Why do I have imbalance? Should I address the problem?**_

- Is it because one class is more rare than the other?
  - If so, you need to determine whether you care about one type of error more than the other
  - If so, it is reasonable to use `class_weight` to address this
- Is it because of my data collection methods?
  - If so, it suggests deployment and training data come from different distributions
  - Reasonable to use `class_weight` to fix this
- If not to both, it may be fine to ignore the class imbalance.

_**How can I fix the imbalance?**_

- Thresholding
- Modify the class weight (e.g., `class_weight='balanced'` or `class_weight={1:10}`)
  - Generally reduces accuracy
  - Output of `predict_proba` loses some of its meaning if this was changed because you care about one type of error more than the other
  - Shifts all probabilities toward one direction but won't necessarily separate the classes for a given threshold
- _**Stratified splits**_ (`StratifiedKFolds`) allows for cross-validation on folds that preserve the percentage of samples for each class.
  - No longer a random sample (not a big issue)
  - Shouldn't matter as much if many examples
  - Useful in multi-class

### Regression Metrics (Error metrics)

Unlike classification, we cannot just check for equality. To determine whether an error metric is reasonable, consider the error relative to the actual value.

_**What happens when we log transform the targets?**_

- _**MAPE:**_ Log transforming the targets turns addition/subtraction into multiplication/division, which is what we want (fractional error instead of absolute error). In short, it improves MAPE (lower MAPE).
- _**R*2/MSE:**_ It gets worse.
  - Typical because the default settings try to optimize this metric
- Assumes target values are positive
  - If there is a value that equals 0, you can $log(1+y)$ using `log1p` and `expm1`.

One convenient way to log-transform the data is using `TransformedTargetRegressor`, which also allows us to use functions like `cross_validate` when transforming the targets.

#### Mean Squared Error (MSE)

- Perfect predictions have MSE=0
- The goodness of the score depends on the scale of the targets
  - EX. If we work on cents instead of dollars, our MSE would be $100^{2}$ higher.
- Reversible so `mean_squared_error(y_train, preds)` is the same as `mean_squared_error(preds, y_train)`
- By default, if you call `fit`, it minimizes MSE.

#### Root Mean Squared Error (RMSE)

- Since regression has units, it's useful to look at RMSE over MSE because unit is more interpretable than unit^2
- Perfect predictions have RMSE=0

#### Coefficient of determination (R^2)

- Flipped MSE and normalized so the max is 1
- Perfect predictions have R^2=1
- Negative values are worse than `DummyRegressor`
- Not reversible so `r2_score(y_test, dummy.predict(X_test))` is not the same as `r2_score(dummy.predict(X_test), y_test)`
- By default, if you call `fit`, it maximizes R^2.

#### Mean Absolute Percent Error (MAPE)

Calculates the percent prediction error

- Lower is better
- You can ignore negative signs

### Prediction interval (Regression)

Prediction intervals are useful for communicating uncertainty. Unlike classifiction, we do not have `predict_proba` in regression. Instead we use prediction intervals as a measure of confidence. Some variants of intervals are:

- Confidence intervals for parameter estimates
- Confidence intervals for prediction
- Prediction intervals (for predictions, also take into account uncertainty in data)

_**Creating prediction intervals**_

- **Bootstrapping:** A statistically reasonable way to generate a prediction interval that trains your model on different variants of the dataset (sampled with replacement).
- **Min-Max:** Based on an ensemble, getting the min and max values and using that as the interval. Not great since there is no statistical underpinning.
- **Standard deviation:** Based on an ensemble, getting the standard deviation of the predictions and using that as the interval.
- **Quantile regression:** Regression using a different type of error metric. Instead of trying to minimize an error like MSE, we want the true value to be below the predicted value X% of the time. 
  - You can use this to get prediction intervals by training two models at two different quantiles (10% and 90%) using `GradientBoostingRegressor` or `LGBMRegressor`.

### [Feature Importances](https://github.com/UBC-CS/cpsc330/blob/master/lectures/11_feature-importances.ipynb)

Feature importances tell you about how features correlate with predicted prices. Therefore, we cannot make causal statements.

_**Why do we want feature importances?**_

- To identify features that are not useful and maybe remove them
- Get guidance on what new data to collect
  - New features related to useful features -> better results
  - Don't bother collecting useless features -> save resources
- Help explain why the model is making certain predictions
  - Debugging, if the model is behaving strangely
  - Regulatory requirements
  - Fairness/bias
  - Keep in mind this can be used on deployment predictions
- And more...

#### Linear Models

$Prediction = intercept + \sum_{i} coefficient i x feature i$

_**What would cause the prediction to be different than expected?**_
After scaling, the coefficients likely are per unit, but per $X$ units. The prediction Scaling influences the coefficient score. For instance, StandardScaler subtracted the mean and divided by the standard deviation.

_**What happens if we log transform our space?**_
If we increase the feature by 1 scaled unit, then we increase the log predicted price by the exp(coefficient).

#### Non-Linear Models

Two ways of getting feature importances from non-linear models is with sklearn `feature_importances_` and SHAP. As these values are unsigned, they tell us about the importance but not direction. Unlike linear models, increasing a feature for non-linear models may cause the prediction to first go up, then go down.

- If comparing two different models, do not compare actual values but rather order of feature importance.
- `feature_importances_` pertain only to training data while SHAP can apply to all data.

#### Feature Correlation

Feature correlations can be determined using a heatmap. However, it is overly simplified because you can only look at each feature in isolation.

_**So why use it?**_
If two features are highly correlated, we can consider whether we need both values.
But note that if we add/remove a feature, a different feature may become important.

### Feature Selection

Feature selection looks at which features are important for predicting $y$, and removing the features that aren't.

_**Why not use all features?**_

- Data collection: It might be cheaper to collect fewer columns
- Computation: Models fit/predict faster with fewer columns
- Fundamental tradeoff: We might be able to reduce overfitting by removing useless features

_**How to do feature selection?**_

- Domain knowledge
- Importance-based feature elimination

_**Warnings about feature selection**_

- A feature's relevance is only defined in the context of other features
  - Adding/removing features can make features relevant/irrelevant
- If features can be predicted from other features, you cannot know which one to pick
- Relevance for features does not have a causal relationship
- Don't be overconfident
  - The methods we have seen probably do not discover the ground truth and how the world really works
  - They simply tell you which features help in predicting $y_i$ for the data you have

#### Importance-based feature elimination

The basic idea for importance-based feature elimination is to throw away the least important feature

##### Recursive feature elimination (RFE)

1. Decide $k$, the number of features to select.
2. Assign importances to features (by fitting a model and looking at `coef_` or `feature_importances_`)
3. Remove the least important feature
4. Repeat steps 2-3 until only $k$ features are remaining.

_**Why don't we just remove all the less important features at once?**_
By removing a feature, the importances may change so a feature that was once not very important may become more important.

_**How can we determine how many features to select?**_

We can use cross validation with RFECV. However, this violates the Golden Rule.

## Unsupervised Learning

Unsupervised learning looks for patterns in a data set without labels (e.g., Nearest Neighbours).

## Big Datasets

_**Problems with big datasets:**_

- The code is too slow
- The dataset doesn't fit in memory - Can't even load it with `pd.read_csv`

_**Solutions:**_

- Subset your data for experimentation/hyperparameter tuning, then train your final model on the whole dataset (once)
- "SGD" (stochastic gradient descent): `SGDClassifier` and `SGDRegressor`
  - Quickly finding "approximately" the best coefficients when calling `fit`
  - SGDRegressor is basically equivalent to Ridge.
  - SGDRegressor(loss='huber') is basically equivalent to HuberRegressor.
  - SGDClassifier(loss='log') is basically equivalent to LogisticRegression, except the parameter is called alpha instead of C (like Ridge).
  - With other settings they are equivalent to other models, but this is good enough.

_**Why is "SGD" faster?**_
It does a worse job of `fit` (time tradeoff).

_**When should you use "SGD"?**_
On big datasets. If you want to wait a short amount of time, you can use `SGDClassifier`. If you want to wait a long time, just use `LogisticRegression`.

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

## Model Deployment

After we train a model, we want to use it! The user likely does not want to install your Python stack, train your model. You don't necessarily want to share your dataset.

So we need to do two things:

- Save/store your model for later use.
- Make the saved model conveniently accessible

Upon deployment, we'll re-fit the model on the full dataset to get it ready for deployment.

- This is probably a good idea, because more data is better.
- It's also a little scary, because we can't test this new model.

_**How to deploy prediction model**_

1. Develop a RESTful web API that accepts HTTP requests in the form of input data and returns a prediction.
2. Build a web application with a HTML user-interface that interacts directly with our API.

_**Things to consider:**_

- Privacy/security
- Scaling
- Error handling
- Real-time / speed
- Low-resource environments (e.g. edge computing)
- etc.

## Guidelines

- Do train-test split right away and only once
- Don't look at the test set until the end
- Don't call `fit` on test/validation data
- Use pipelines
- Use baselines (e.g., Dummy)

## Resources

- [MDS Terminology](https://ubc-mds.github.io/resources_pages/terminology/)
- Pre-Midterm Cheatsheet (found in local workspace)
