from sklearn.dummy import DummyRegressor

dummy = DummyRegressor()
pd.DataFrame(cross_validate(dummy, X_train, y_train, return_train_score=True))
dummy.fit(X_train, y_train)
dummy.predict(X_train) == y_train