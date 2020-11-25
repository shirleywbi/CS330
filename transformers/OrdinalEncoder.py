from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(dtype=int)
transformed = oe.fit_transform(X_train_census[categorical_features])
transformed = pd.DataFrame(data=transformed, columns=categorical_features, index=X_train_census.index)