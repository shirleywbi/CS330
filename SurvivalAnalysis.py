# How to handle Censoring
## Approach 1: Consider the cases for which we have the time
df_train_churn = df_train.query("Churn == 'Yes'")
df_test_churn = df_test.query("Churn == 'Yes'")

preprocessing_notenure = ColumnTransformer([
    ('scale',  make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features[1:]),
    ('ohe',    OneHotEncoder(), categorical_features) 
])

tenure_lm = make_pipeline(preprocessing_notenure, Ridge())
tenure_lm.fit(df_train_churn.drop(columns=["tenure"]), df_train_churn["tenure"])
tenure_lm.predict(df_test_churn.drop(columns=["tenure"]))[:10]

## Approach 2: Assume everyone churns right now
tenure_lm.fit(df_train.drop(columns=["tenure"]), df_train["tenure"])
tenure_lm.predict(df_test.drop(columns=["tenure"]))[:10]



# Kaplan-Meier survival curve
kmf = lifelines.KaplanMeierFitter()
kmf.fit(df_train_surv["tenure"], df_train_surv["Churn"])

kmf.survival_function_.plot()  # kmf.plot() for error bars
plt.title('Survival function of customer churn')
plt.xlabel('Time with service (months)')
plt.ylabel('Survival probability')

## K-M curve for subgroups
T = df_train_surv["tenure"]
E = df_train_surv["Churn"]
senior = df_train_surv["SeniorCitizen_1"] == 1
ax = plt.subplot(111)

kmf.fit(T[senior], event_observed=E[senior], label="Senior Citizens")
kmf.plot(ax=ax)

kmf.fit(T[~senior], event_observed=E[~senior], label="Non-Senior Citizens")
kmf.plot(ax=ax)

plt.ylim(0, 1)
plt.xlabel('Time with service (months)')
plt.ylabel('Survival probability')



# Cox proportional hazards model
cph = lifelines.CoxPHFitter(penalizer=0.1)  # penalizer takes it from LR to Ridge
cph.fit(df_train_surv, duration_col='tenure', event_col='Churn')

cph_params = pd.DataFrame(cph.params_)
cph_params.sort_values(by="coef", ascending=False)

cph.summary # gives you confidence intervals, etc.

## Plot examples
cph.plot_partial_effects_on_outcome('Contract_Two year', [0, 1])
cph.plot_partial_effects_on_outcome('MonthlyCharges', [10, 100, 1000, 10_000])

## Prediction
cph.predict_expectation(df_test_surv).head() # assumes they just joined right now
cph.predict_expectation(df_test_surv, conditional_after=df_test_surv["tenure"]).head() # conditional_after indicates otherwise

cph.predict_survival_function(df_test_surv[:5]).plot()
plt.xlabel('Time with service (months)')
plt.ylabel('Survival probability')

## Measures of model accuracy
cph.score(df_train_surv) # partial log likelihood
cph.score(df_train_surv, scoring_method="concordance_index") # concordance index
