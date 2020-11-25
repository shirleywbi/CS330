from sklearn.pipeline import Pipeline, make_pipeline

numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                    'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
ordinal_features = ['education']
target_column = 'income'

countvec = CountVectorizer(min_df=50, binary=True)
lr = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ('countvec', countvec),
    ('lr', lr)])