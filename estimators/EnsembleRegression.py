from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor

# Averaging
regressors = {
    'linear regression' : make_pipeline(preprocessing, Ridge()),
    'decision tree' : make_pipeline(preprocessing, DecisionTreeRegressor()),
    'random forest' : make_pipeline(preprocessing, RandomForestRegressor(n_estimators=10, random_state=999)),
    'XGBoost' : make_pipeline(preprocessing, XGBRegressor(random_state=999)) 
}

results_dict = {name: cross_validate_std(regressor, X_train, y_train, return_train_score=True) for name, regressor in regressors.items()}
results_df = pd.DataFrame(results_dict).T.sort_values(by=["test_score"], ascending=False)

averaging_model = VotingRegressor(list(regressors.items()))
averaging_model.fit(X_train, y_train)
cross_validate_std(averaging_model, X_train, y_train, return_train_score=True)
averaging_model.predict(X_test[:1])[0]

r = {name : regressor.predict(X_test[:1])[0] for name, regressor in averaging_model.named_estimators_.items()}
r = pd.DataFrame(r, index=["Prediction"]).T

# Stacking
stacking_model = StackingRegressor(list(regressors.items()))
stacking_model.fit(X_train, y_train)
cross_validate_std(stacking_model, X_train, y_train, return_train_score=True)
pd.DataFrame(data=stacking_model.final_estimator_.coef_, index=regressors.keys(), columns=["Coefficient"])