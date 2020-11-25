# CPSC 330 Summary Sheet

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

Overfitting occurs when the model is too close to the model. Positively correlated with complexity.

_**Occurs when:**_

- $E_{train} < E_{best} < E_{test}$
- Train score is much better than validation/test score

### Underfitting

Underfitting occurs when the model is so simple that it does not pick up the patterns from the training set.

_**Occurs when:**_

- $E_{best} < E_{train} ~< E_{test}$
- Train score is poor, and is similar to validation/test score

### TODO: Determining performance

Compare against Dummy

## Classifiers

_**What does .score() do?**_
It calls predict then compares the predictions to the true labels.

_**Why would score not result in 100% accuracy?**_
There may be instances of duplicate features with different target values.

### DummyClassifier

DummyClassifier makes predictions based on the target value, ignoring the features. Any reasonable classifier should have a higher prediction accuracy than `DummyClassifier`.

- May not always get >= 50% prediction accuracy (e.g., when there are more than 2 target values)

### Decision Trees

Decision trees are classifiers that make predictions by sequentially looking at features and checking whether they are above/below a threshold. They learn axis-aligned decision boundaries (vertical and horizontal lines with 2 features).

- Categorical - yes/no questions
- Numeric - thresholds (e.g., 5%, 50%)

_**Why not use a very deep tree?**_
Although it will result in larger accuracy of the training data, it may result in overfitting and not apply to the test/deployment data.

### Logistic Regression (Linear Classifier)

Logistic Regression is a popular linear classifier that consists of input features, coefficients (weights) per feature, and bias or intercept.

_**Advantages:**_

- Popular
- Fast training and testing (e.g., huge datasets)
- Interpretability
  - Coefficients (weights) are how much a given feature changes the prediction and in what direction

_**predict_proba**_
Unlike `predict` that outputs the value with the highest confidence score, `predict_proba` outputs a list of confidence scores on a set of given features.

_**C**_
C is correlated to the complexity of a model. Smaller C leads to less confident predictions (probabilities closer to 0.5).

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

### CountVectorizer

Transforms text data into counts or present/absent.

## Pipelines

Sequentially apply a list of transforms and a final estimator, refitting again on each fold.

_**Advantages:**_

- Cleanly call `fit` on train split and `score` on test split
- Can't accidentally re-fit the preprocessor on the test data
- Automatically makes sure the same transformations are applied to train and test

![pipeline](https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/images/pipeline.png)
[Source](https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/images/pipeline.png)

_________________________________
## Unsupervised Learning

TODO

## Resources

- [MDS Terminology](https://ubc-mds.github.io/resources_pages/terminology/)