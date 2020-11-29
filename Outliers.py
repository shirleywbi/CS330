# Detecting outliers
## Histogram
census_train["capital.gain"].hist()

## Scatter plot
housing_train.plot.scatter(x="LotArea", y="SalePrice")


## QuantileTransformer
qt = QuantileTransformer()
area_transformed_qt = qt.fit_transform(housing_train[["LotArea"]])
plt.hist(area_transformed_qt, bins=100)


## Isolation Forests
### Preprocessing
numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                    'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
target_column = 'income'
pipe_cat = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
census_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', pipe_cat, categorical_features)
])
census_preprocessor.fit(census_train)

census_columns = numeric_features + list(census_preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names(categorical_features))
census_train_enc = pd.DataFrame(census_preprocessor.transform(census_train), index=census_train.index, columns=census_columns)

### Using isolation forest
isolation = IsolationForest(random_state=123)
predicted_outliers = isolation.fit_predict(census_train_enc)
predicted_outliers # returns array: +1 for inlier and -1 for outlier


## HuberRegressor
from sklearn.linear_model import HuberRegressor

hr = HuberRegressor(max_iter=1000, epsilon=0.35) # smaller epsilon = more robust
hr.fit(X_train_enc, y_train_corrupted)
hr.score(X_test_enc, y_test)
hr.outliers_ # Array of true/false to indicate whether its an outlier


## Random Forest
rf_uncorrupted = RandomForestRegressor(max_depth=20, n_estimators=20, random_state=333)
rf_uncorrupted.fit(X_train_enc, y_train)