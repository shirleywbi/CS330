# https://github.com/UBC-CS/cpsc330/blob/master/lectures/17_time-series.ipynb
from sklearn.model_selection import TimeSeriesSplit

def preprocess_features(df_train, df_test, numeric_features, categorical_features, drop_features):

    all_features = set(numeric_features + categorical_features + drop_features)
    if set(df_train.columns) != all_features:
        print("Missing columns", set(df_train.columns) - all_features)
        print("Extra columns", all_features - set(df_train.columns))
        raise Exception("Columns do not match")
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])  
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='?')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    preprocessor.fit(df_train);

    ohe = preprocessor.named_transformers_['categorical'].named_steps['onehot']
    ohe_feature_names = list(ohe.get_feature_names(categorical_features))
    new_columns = numeric_features + ohe_feature_names

    X_train_enc = pd.DataFrame(preprocessor.transform(df_train).toarray(), index=df_train.index, columns=new_columns)
    X_test_enc  = pd.DataFrame(preprocessor.transform(df_test).toarray(),  index=df_test.index,  columns=new_columns)
    
    y_train = df_train["RainTomorrow"]
    y_test  = df_test["RainTomorrow"]
    
    return X_train_enc, y_train, X_test_enc, y_test, preprocessor

# Splitting data
df_rain = pd.read_csv("data/weatherAUS.csv", parse_dates=["Date"])
df_train = df_rain.query('Date <= 20150630')
df_test  = df_rain.query('Date >  20150630')

df_train_sort = df_train.query("Location == 'Sydney'").sort_values(by="Date")
df_test_sort = df_test.query("Location == 'Sydney'").sort_values(by="Date")

plt.plot(df_train_sort["Date"], df_train_sort["Rainfall"], 'b', label='train')
plt.plot(df_test_sort["Date"], df_test_sort["Rainfall"], 'r', label='test')
plt.xticks(rotation='vertical')
plt.legend()
plt.ylabel("Rainfall (mm)")
plt.title("Train/test rainfall in Sydney")

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features, categorical_features, drop_features)

lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
lr_pipe.fit(df_train, y_train)
lr_pipe.score(df_train, y_train)
lr_pipe.score(df_test, y_test)

lr_coef = pd.DataFrame(data=np.squeeze(lr_pipe[1].coef_), index=X_train_enc.columns, columns=["Coef"])
lr_coef.sort_values(by="Coef", ascending=False)

## TimeSeries Split (Cross-validation)
from sklearn.model_selection import TimeSeriesSplit
df_train_ordered = df_train.sort_values(by=["Date"])
y_train_ordered = df_train_ordered["RainTomorrow"]
cross_val_score(lr_pipe, df_train_ordered, y_train_ordered, cv=TimeSeriesSplit()).mean()

## Shuffle Split (Cross-validation)
from sklearn.model_selection import ShuffleSplit
cross_val_score(lr_pipe, df_train, y_train, cv=ShuffleSplit()).mean()



# Encoding date/time as features
## Encoding time as number
first_day = df_train["Date"].min()
df_train = df_train.assign(Days_since=df_train["Date"].apply(lambda x: (x-first_day).days))
df_test = df_test.assign(Days_since=df_test["Date"].apply(lambda x: (x-first_day).days))

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features  + ["Days_since"], 
        categorical_features, 
        drop_features)

## One-hot encoding of the month
df_train = df_rain.query('Date <= 20150630')
df_test  = df_rain.query('Date >  20150630')
df_train = df_train.assign(Month=df_train["Date"].apply(lambda x: x.month)) # x.month_name() to get the actual string
df_test  = df_test.assign( Month=df_test[ "Date"].apply(lambda x: x.month))

## One-hot encoding seasons
WINTER_MONTHS = {5,6,7,8,9}
df_train = df_train.assign(Winter=df_train["Month"].isin(WINTER_MONTHS))
df_test  = df_test.assign( Winter=df_test[ "Month"].isin(WINTER_MONTHS))

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features + ["Winter"], 
        categorical_features, 
        drop_features + ["Month"]) # Note that month was dropped

lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

## Periodic Encoding
WINTER_MONTHS = {5,6,7,8,9}
df_train = df_train.assign(Winter=df_train["Month"].isin(WINTER_MONTHS))
df_test  = df_test.assign( Winter=df_test[ "Month"].isin(WINTER_MONTHS))

df_train = df_train.assign(Month_sin = np.sin(2*np.pi*df_train["Month"]/12))
df_train = df_train.assign(Month_cos = np.cos(2*np.pi*df_train["Month"]/12))

df_test = df_test.assign(Month_sin = np.sin(2*np.pi*df_test["Month"]/12))
df_test = df_test.assign(Month_cos = np.cos(2*np.pi*df_test["Month"]/12))

month = np.arange(1,13)
enc_sin = np.sin(2*np.pi*month/12)
enc_cos = np.cos(2*np.pi*month/12)
plt.plot(month,enc_sin)
plt.plot(month,enc_cos, 'r')

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features + ["Month_sin", "Month_cos"], 
        categorical_features, 
        drop_features + ["Month", "Winter"])


## EXAMPLE (Complex)
### Encoding average monthly rainfall
monthly_avg_rainfall = df_train.groupby("Month")["Rainfall"].mean()
df_train = df_train.assign(Monthly_rainfall = df_train["Date"].apply(lambda x: monthly_avg_rainfall[x.month]))
df_test = df_test.assign(Monthly_rainfall = df_test["Date"].apply(lambda x: monthly_avg_rainfall[x.month]))

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features + ["Monthly_rainfall"], 
        categorical_features, 
        drop_features + ["Month", "Winter", "Month_sin", "Month_cos"])

lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))



# Lag Features
def create_lag_feature(df, orig_feature, lag):
    """Creates a new df with a new feature that's a lagged version of the original, where lag is an int."""
    # note: pandas .shift() kind of does this for you already, but oh well I already wrote this code
    
    new_df = df.copy()
    new_feature_name = "%s_lag%d" % (orig_feature, lag)
    new_df[new_feature_name] = np.nan
    for location, df_location in new_df.groupby("Location"): # Each location is its own time series
        new_df.loc[df_location.index[lag:],new_feature_name] = df_location.iloc[:-lag][orig_feature].values
    return new_df

# Assume df was sorted by location and date for both features and targets before preprocessing
df_rain_modified = create_lag_feature(df_rain, "Rainfall", 1)

df_train = df_rain_modified.query('Date <= 20150630')
df_test  = df_rain_modified.query('Date >  20150630')

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(df_train, df_test, 
        numeric_features + ["Rainfall_lag1"], 
        categorical_features, 
        drop_features)

lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
lr_pipe.fit(df_train, y_train)
lr_pipe.score(df_train, y_train)

df_rain_modified = create_lag_feature(df_rain, "Rainfall", 1)
df_rain_modified = create_lag_feature(df_rain_modified, "Rainfall", 2)
df_rain_modified = create_lag_feature(df_rain_modified, "Rainfall", 3)
df_rain_modified = create_lag_feature(df_rain_modified, "Humidity3pm", 1)



# Forecasting further into the future
## Creating a dataset with future data
retail_df = pd.read_csv('data/retail_sales_timeseries.csv', parse_dates=["DATE"])
retail_df.columns=["date", "sales"]
retail_df_train = retail_df.query('date <= 20170101')
retail_df_test  = retail_df.query('date >  20170101')

def lag_df(df, lag, cols):
    return df.assign(**{f"{col}-{n}": df[col].shift(n) for n in range(1, lag + 1) for col in cols})

retail_lag_5 = lag_df(retail_df, 5, ["sales"])
retail_train_5 = retail_lag_5.query('date <= 20170101')
retail_test_5  = retail_lag_5.query('date >  20170101')
retail_train_5 = retail_train_5[5:].drop(columns=["date"])
retail_train_5_X = retail_train_5.drop(columns=["sales"])
retail_train_5_y = retail_train_5["sales"]

## Analyzing that dataset
from sklearn.ensemble import RandomForestRegressor
retail_model = RandomForestRegressor()
retail_model.fit(retail_train_5_X, retail_train_5_y)
preds = retail_model.predict(retail_test_5.drop(columns=["date", "sales"]))
retail_test_5_preds = retail_test_5.assign(predicted_sales = preds)

# Trends
retail_train_5_date = retail_lag_5.query('date <= 20170101')
first_day_retail = retail_train_5_date["date"].min()

retail_train_5_date.assign(Days_since=retail_train_5_date["date"].apply(lambda x: (x-first_day_retail).days))
