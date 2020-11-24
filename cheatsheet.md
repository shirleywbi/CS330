# CPSC 330 Summary Sheet
Based on [UBC CPSC 330 Lectures](https://github.com/UBC-CS/cpsc330) by Mike Gelbert

## Supervised Learning

Given a set of features $X$, predict associated targets $y$.

There are two types of models:

- **Classification** = categorical target
- **Regression** = quantitative target

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

### Overfitting

Because the goal of supervised learning is to predict unseen/new data, we do not want the model to be too closely modeled to our training data. As a result, we split the data into **training** and **testing**.

## Unsupervised Learning

TODO

## Resources

- [MDS Terminology](https://ubc-mds.github.io/resources_pages/terminology/)