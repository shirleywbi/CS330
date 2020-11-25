from constants import X_train, y_train

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# cross_val_score does not shuffle the data

## cross_val_score
tree = DecisionTreeClassifier(max_depth=1)
cv_score = cross_val_score(tree, X_train, y_train, cv=5)
print(f"Average cross-validation score = {np.mean(cv_score):.2f}")

## cross_validate
cross_validate(tree, X_train, y_train, cv=10, return_train_score=True)


# AUTOMATIC HYPERPARAMETER OPTIMIZATION
## Exhaustive Grid Search
param_grid = {
    "countvec__min_df" : [0, 10, 100],
    "lr__C" : [0.01, 1, 10, 100]
}
lr = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(pipe, param_grid, verbose=2, n_jobs=-1)
grid_search.fit(X_train_imdb_raw, y_train_imdb) # NOTE: Pass raw data with Pipeline

grid_search.best_params_
grid_search.best_score_
grid_search.best_estimator_

## Randomized Hyperparamter Optimization
import scipy.stats

### Different ways of representing parameter choices
param_choices = {
    "countvec__min_df" : np.arange(0,100),
    "lr__C" : 2.0**np.arange(-5,5)
}
param_choices = {
    "countvec__min_df" : scipy.stats.randint(low=0, high=300),
    "lr__C" : scipy.stats.randint(low=0, high=300) # TODO: this is lame, pick a continuous prob dist
}

random_search = RandomizedSearchCV(pipe, param_choices,
    n_iter = 12, 
    verbose = 1,
    n_jobs = -1,
    random_state = 123)
random_search.fit(X_train_imdb_raw, y_train_imdb)

random_search.best_params_
random_search.best_score_
random_search.best_estimator_