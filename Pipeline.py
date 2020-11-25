from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                    'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
ordinal_features = ['education']
target_column = 'income'

pipe_cat = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('cat', pipe_cat, categorical_features),
    ('num', StandardScaler(), numeric_features)
])

pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train_nan, y_train)
pipe.predict(X_test_nan)
cross_validate(pipe, X_train_nan, y_train)