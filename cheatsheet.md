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

### Classifiers

_**What does .score() do?**_
It calls predict then compares the predictions to the true labels.

_**Why would score not result in 100% accuracy?**_
There may be instances of duplicate features with different target values.

#### DummyClassifier

DummyClassifier makes predictions based on the target value, ignoring the features. Any reasonable classifier should have a higher prediction accuracy than `DummyClassifier`.

- May not always get >= 50% prediction accuracy (e.g., when there are more than 2 target values)

#### Decision Trees

Decision trees are classifiers that make predictions by sequentially looking at features and checking whether they are above/below a threshold. They learn axis-aligned decision boundaries (vertical and horizontal lines with 2 features).

- Categorical - yes/no questions
- Numeric - thresholds (e.g., 5%, 50%)

_**Why not use a very deep tree?**_
Although it will result in larger accuracy of the training data, it may result in overfitting and not apply to the test/deployment data.

### Sources of Error

Because the goal of supervised learning is to predict unseen/new data, we do not want the model to be too closely modeled to our training data. As a result, we split the data into **training** and **testing**.

- Data should be split randomly

_**Why won't you ever have $E_{best} = 0$?**_
Because the target is not a fixed deterministic function of the features. There is always some randomness.

#### Fundamental Trade-off

As you increase model complexity, $E_{train}$ tends to go down but $E_{test} - E_{train}$ tends to go up.

#### Overfitting

Overfitting occurs when the model is too close to the model. Positively correlated with complexity.

_**Occurs when:**_

- $E_{train} < E_{best} < E_{test}$
- Train score is much better than validation/test score

#### Underfitting

Underfitting occurs when the model is so simple that it does not pick up the patterns from the training set.

_**Occurs when:**_

- $E_{best} < E_{train} ~< E_{test}$
- Train score is poor, and is similar to validation/test score

### Hyperparameter Selection (Cross-validation)

To tune our hyperparameters, we need to add an additional split for validation so that our test data remains unseen (to simulate deployment data). In total, we have 4 datasets: train, validation, test and deployment.

||fit|score|predict|
|-|-|-|-|
|Train      |✔️ |✔️  |✔️|
|Validation |   |✔️  |✔️|
|Test       |   |once|once|
|Deployment |   |    |✔️|

#### $k$-fold cross-validation

To obtain a better estimate of test error than just using a single validation set by using multiple splits on all the data. Each fold turns into a validation set and the values for each fold are averaged.

_**Advantage**_

- More accurate score estimates, thus better models
- Gives a measure of uncertainty in the scores

_**Disadvantage**_

- Speed
- More complicated code

_**Finding the optimal model**_
Pick the model with the lowest validation error (reasonable for now) and low approximation error (if possible)

## Unsupervised Learning

TODO

## Resources

- [MDS Terminology](https://ubc-mds.github.io/resources_pages/terminology/)