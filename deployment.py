# https://github.com/UBC-CS/cpsc330/blob/master/lectures/24_deployment-conclusion.ipynb
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate

abalone_df = pd.read_csv('data/abalone.csv',
                       names = ['sex', 'length', 'diameter', 'height',
                                'whole_weight', 'shucked_weight', 'viscera_weight',
                                'shell_weight', 'rings'])

features = ['length', 'diameter', 'height', 'whole_weight']

X = abalone_df[features]
y = abalone_df['rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

model = RandomForestRegressor(n_estimators=10, random_state=123)
model.fit(X_train, y_train)

with open('web_api/abalone_predictor.joblib', 'wb') as f:
    joblib.dump(model, f)
with open('web_application/abalone_predictor.joblib', 'wb') as f:
    joblib.dump(model, f) # Dump regressor into file f


def return_prediction(model, input_json):
    
    input_data = [[input_json[k] for k in features]]
    prediction = model.predict(input_data)[0]
    
    return prediction

example_input_json = {
    'length': 0.41,
    'diameter': 0.33,
    'height': 0.10,
    'whole_weight': 0.36
}

return_prediction(model, example_input_json)