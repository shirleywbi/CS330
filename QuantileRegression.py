from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline, make_pipeline

lgbm = make_pipeline(preprocessing, LGBMRegressor())
lgbm.fit(X_train, y_train)

lgbm_90 = make_pipeline(preprocessing, LGBMRegressor(objective='quantile', alpha=0.9)).fit(X_train, y_train)
lgbm_50 = make_pipeline(preprocessing, LGBMRegressor(objective='quantile', alpha=0.5)).fit(X_train, y_train)
lgbm_10 = make_pipeline(preprocessing, LGBMRegressor(objective='quantile', alpha=0.1)).fit(X_train, y_train)

lgbm_90.predict(X_test[:1])
lgbm_50.predict(X_test[:1])
lgbm_10.predict(X_test[:1])
