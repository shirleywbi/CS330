X = [[0, 0]]
y = 'blue'

from sklearn.dummy import DummyClassifier

dc = DummyClassifier(strategy='prior')
dc.fit(X, y)
dc.score(X, y)
dc.predict(X)
