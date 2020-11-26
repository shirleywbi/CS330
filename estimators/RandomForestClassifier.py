from sklearn.ensemble import RandomForestClassifier

"""
n_estimators: number of decision trees (higher = more complexity)
max_depth: max depth of each decision tree (higher = more complexity)
max_features: the number of features you get to look at each split (higher = more complexity)
"""

# Simple example
rf = RandomForestClassifier(class_weight='balanced', random_state=999) # scaling not needed
rf_results = pd.DataFrame(cross_validate(rf, X_train, y_train, scoring=score_method, return_train_score=True)).mean()

# Multiple example
rf_demo = RandomForestClassifier(max_depth=2, n_estimators=3, random_state=321)
rf_demo.fit(X_train, y_train)
for i, tree in enumerate(rf_demo.estimators_):
    print("\n\nTree", i+1)
    display(display_tree(X_train.columns, tree))