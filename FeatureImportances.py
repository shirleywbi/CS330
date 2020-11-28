lr = make_pipeline(preprocessing, Ridge())
lr.fit(X_train, y_train)

# Feature Importances
lr_coefs = pd.DataFrame(data=lr[1].coef_, index=new_columns, columns=["Coefficient"])
lr_coefs.head(30)

lr[1].intercept_
lr_coefs.loc[["PoolArea", "LotFrontage"]] # To get specific column

# Feature Correlations (with Heatmap)
import seaborn as sns

## Example 1
sns.heatmap(X_train_enc[numeric_features[:5]].corr())

## Example 2
plt.figure(figsize=(15,15))
sns.set(font_scale=1)
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues);

# Checking predictions
one_example = X_test[:1]
one_example_perturbed = one_example.copy()
one_example_perturbed["LotArea"] += 1
lr.predict(one_example_perturbed) - lr.predict(one_example)

lr_coefs.loc[["LotArea"]]


# Coefficient Interpretation
## Method 1: Handling with Scaling

scaler = preprocessing.named_transformers_['numeric']['standardscaler']
lr_scales = pd.DataFrame(data=np.sqrt(scaler.var_), index=numeric_features, columns=["Scale"])
lr_coefs.loc["LotArea","Coefficient"]/lr_scales.loc["LotArea","Scale"]

## Method 2: With Inverse Transform
delta = scaler.inverse_transform(np.ones(len(numeric_features))) - scaler.inverse_transform(np.zeros(len(numeric_features)))
lr_scales2 = pd.DataFrame(data=delta, index=numeric_features, columns=["Scale"])
lr_scales2.tail()

## Method 3: Log-Transform
lr_log = make_pipeline(preprocessing, Ridge())
lr_log.fit(X_train, np.log(y_train))
lr_log_coefs = pd.DataFrame(data=lr_log[1].coef_, index=new_columns, columns=["Coefficient"])
np.exp(lr_log_coefs.loc["LotArea", "Coefficient"])
np.exp(lr_log_coefs.loc["LotFrontage", "Coefficient"])


# Non-linear feature importances
## feature_importances_
rf = make_pipeline(preprocessing, RandomForestRegressor(random_state=111))
rf.fit(X_train, np.log(y_train))

rf_importances = pd.DataFrame(data=rf[1].feature_importances_, index=new_columns, columns=["Importance"])
rf_importances.sort_values(by="Importance", ascending=False).head()

## SHAP
explainer = shap.TreeExplainer(rf[1])
X_train_enc = pd.DataFrame(preprocessing.transform(X_train), index=X_train.index, columns=new_columns)

shap_values = explainer.shap_values(X_train_enc)
shap.initjs()

## Different plots
### SHAP represents feature importance
shap.force_plot(explainer.expected_value, shap_values[0], X_train_enc.iloc[0])
shap.dependence_plot("GrLivArea", shap_values, X_train_enc)
shap.summary_plot(shap_values, X_train_enc, plot_type="bar")