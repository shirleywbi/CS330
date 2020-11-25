from sklearn.preprocessing import OneHotEncoder

ohe_def = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)
ohe_def.fit(X_train_census[["relationship"]])
pd.DataFrame(data=ohe_def.transform(X_train_census[["relationship"]]), columns=ohe_def.get_feature_names(["relationship"]), index=X_train_census.index)

# Hard-coding all categories
all_countries = census["native.country"].unique()
ohe_cat = OneHotEncoder(categories=all_countries)