from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Ridge
lr = make_pipeline(preprocessing, Ridge(alpha=1))
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_preds.max()
lr_preds.min()

# Mean squared error (MSE)
preds = dummy.predict(X_train)
mean_squared_error(y_train, preds)

# Root mean squared error (RMSE)
np.sqrt(mean_squared_error(y_train, lr_tuned.predict(X_train)))

# R^2 score
r2_score(y_train, preds)

# Mean Absolute Percent Error (MAPE)
def mape(true, pred):
    return 100.*np.mean(np.abs((pred - true)/true))

pred_train = lr_tuned.predict(X_train)
mape(y_train, pred_train)

## Log transforming targets to reduce MAPE
lr_tuned_log = make_pipeline(preprocessing, Ridge(alpha=best_alpha))
lr_tuned_log.fit(X_train, np.log(y_train))
preds = np.exp(lr_tuned_log.predict(X_train)) # Convert back to unit


# Convenient method to transform and untransform data
from sklearn.compose import TransformedTargetRegressor

ttr = TransformedTargetRegressor(Ridge(alpha=best_alpha), func=np.log1p, inverse_func=np.expm1)
ttr_pipe = make_pipeline(preprocessing, ttr)

ttr_pipe.fit(X_train, y_train); # y_train automatically transformed
ttr_pipe.predict(X_train) # predictions automatically un-transformed