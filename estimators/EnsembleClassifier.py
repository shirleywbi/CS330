from sklearn.ensemble import VotingClassifier, StackingClassifier

classifiers = {
    'logistic regression' : pipe_lr,
    'decision tree' : dt,
    'random forest' : rf,
    'XGBoost' : XGBClassifier(scale_pos_weight=500, random_state=999), 
    'LightGBM' : LGBMClassifier(class_weight='balanced', random_state=999),
    'CatBoost' : CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=999)
}

# Averaging
averaging_model = VotingClassifier(list(classifiers.items()), voting='soft') # need the list() here for cross_val to work!
averaging_model.fit(X_train, y_train) # fit required (instead of passing pre-fit models)

## Looking at a specific test case
t = np.where(y_test)[0][1]
averaging_model.predict(X_test.iloc[t:t+1])
r1 = {name : classifier.predict(X_test[t:t+1])[0] for name, classifier in averaging_model.named_estimators_.items()} # hard voting results
r2 = {name : classifier.predict_proba(X_test[t:t+1])[0] for name, classifier in averaging_model.named_estimators_.items()} # soft voting results

averaging_model.predict_proba(X_test.iloc[t:t+1]) # outputs matrix with average scores


# Stacking
stacking_model = StackingClassifier(list(classifiers_nocat.items()))
stacking_model.fit(X_train, y_train)

## Looking at a specific test case
t = np.where(y_test)[0][1]
stacking_model.predict(X_test[t:t+1])
r3 = {name : classifier.predict_proba(X_test[t:t+1])[0] for name, classifier in stacking_model.named_estimators_.items()}