# Recursive feature elimination (RFE)
## Note: This does not work well with Pipeline or TransformedTargetRegressor

from sklearn.feature_selection import RFE

y_train_log = np.log(y_train)
y_test_log = np.log(y_test)


lr = Ridge(alpha=100)
lr.fit(X_train_enc, y_train_log)

lr.score(X_train_enc, y_train_log)
lr.score(X_test_enc, y_test_log)

coef_df = pd.DataFrame(data=np.abs(lr.coef_), index=X_train_enc.columns, columns=["Coefficients"]).sort_values(by="Coefficients", ascending=False)

rfe = RFE(lr, n_features_to_select=30)
rfe.fit(X_train_enc, y_train_log)
rfe.score(X_train_enc, y_train_log)
rfe.score(X_test_enc, y_test_log)

selected_columns = X_train_enc.columns[rfe.support_] # To see the top 30 features
rfe.ranking_ # To see ranking (may be different from ranked feature importances)

# How to determine # of features to select
## Violates golden rule
lr = Ridge(alpha=100)

rfe_cv = RFECV(lr)
rfe_cv.fit(X_train_enc, y_train_log)
print('Number of selected features: %d/%d' % (rfe_cv.n_features_, X_train_enc.shape[1]))
# print('Feature mask:', rfe_cv.support_)