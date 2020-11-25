from sklearn.pipeline import Pipeline, make_pipeline

countvec = CountVectorizer(min_df=50, binary=True)
lr = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ('countvec', countvec),
    ('lr', lr)])