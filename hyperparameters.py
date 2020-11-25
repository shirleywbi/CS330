from constants import X_train, y_train

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
# cross_val_score does not shuffle the data

# cross_val_score
tree = DecisionTreeClassifier(max_depth=1)
cv_score = cross_val_score(tree, X_train, y_train, cv=5)
print(f"Average cross-validation score = {np.mean(cv_score):.2f}")

# cross_validate
cross_validate(tree, X_train, y_train, cv=10, return_train_score=True)

