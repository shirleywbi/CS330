from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_train_imdb, y_train_imdb)
lr.score(X_train_imdb, y_train_imdb)
lr.score(X_test_imdb, y_test_imdb)

# Predicting Probabilities
## predict: returns highest probability
highest_prob = lr.predict(X_test_imdb)

## predict_proba: returns probability of class 0 and probability of class 1
probs = lr.predict_proba(X_test_imdb)


# Retrieving features
## most positive feature
np.max(lr.predict_proba(X_test_imdb)[:,1]) # score
most_positive_ind = np.argmax(lr.predict_proba(X_test_imdb)[:,1]) # index of score
print(X_test_imdb_raw.iloc[most_positive_ind]) # text

## most negative feature
np.min(lr.predict_proba(X_test_imdb)[:,1])
most_negative_ind = np.argmax(lr.predict_proba(X_test_imdb)[:,0])
print(X_test_imdb_raw.iloc[most_negative_ind])

# Get Coefficients (weights)
## The more positive the sum of all the weights is, the closer to 1 the predicted probability would be.
## The more negative the sum of all the weights is, the closer to 0 the predicted probability would be
## If the sum of all the weights == 0, the predicted probability would be exactly 0.5
vocab = vec.get_feature_names()
weights = lr.coef_.ravel()
words_weights_df = pd.DataFrame(data=weights, index=vocab, columns=['Weight'])
words_weights_df.sort_values(by="Weight", ascending=False)