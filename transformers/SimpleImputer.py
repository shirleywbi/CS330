from sklearn.impute import SimpleImputer

numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                    'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
target_column = 'income'

imp = SimpleImputer(strategy='most_frequent')
imp.fit(X_train_nan[categorical_features])

X_train_imp_cat = pd.DataFrame(imp.transform(X_train_nan[categorical_features]),
                           columns=categorical_features, index=X_train_nan.index)
X_test_imp_cat = pd.DataFrame(imp.transform(X_test_nan[categorical_features]),
                           columns=categorical_features, index=X_test_nan.index)

X_train_imp = X_train_nan.copy()
X_train_imp.update(X_train_imp_cat)

X_test_imp = X_test_nan.copy()
X_test_imp.update(X_test_imp_cat)