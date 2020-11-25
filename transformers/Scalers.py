from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_imp[numeric_features])
scaler.transform(X_train_imp[numeric_features])

scaled_train_df = pd.DataFrame(scaler.transform(X_train_imp[numeric_features]),
    columns=numeric_features, index=X_train_imp.index)

## Checking values
scaled_train_df.mean(axis=0)
scaled_train_df.std(axis=0)

scaled_test_df.mean(axis=0)
scaled_test_df.std(axis=0)


# MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(X_train_imp[numeric_features])
normalized_train = minmax.transform(X_train_imp[numeric_features])
normalized_test = minmax.transform(X_test_imp[numeric_features])

# Checking values
normalized_train.min(axis=0)
normalized_train.max(axis=0)

normalized_test.min(axis=0)
normalized_test.max(axis=0)