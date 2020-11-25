import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from plot_classifier import plot_classifier



# How to read a csv file
df = pd.read_csv('file.csv', index_col=0)
df.head()

df.columns = ['meat', 'grade', 'cilantro']

target_value = 'SalePrice'
X_train = df_train.drop(columns=[target_value])
y_train = df_train[target_value]

X_test = df_test.drop(columns=[target_value])
y_test = df_test[target_value]

# Exploratory
df.describe()
X.shape
X.ndim
y.value_counts()

from pandas_profiling import ProfileReport
profile = ProfileReport(census_train, title='Pandas Profiling Report')
profile.to_notebook_iframe()

## Scatterplot
scatter = plt.scatter(df["meat"], df["grade"], c=df["cilantro"]=="Yes", cmap=plt.cm.coolwarm);
plt.xlabel("Meat consumption (% days)")
plt.ylabel("Expected grade (%)")
plt.legend(scatter.legend_elements()[0], ["No", "Yes"])

scatter.legend_elements()[0]

# Splitting the data (randomly)
## Method: 1
df_train, df_test = train_test_split(df)

## Method: 2
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Taking a subset of a dataset (for speed)
imdb_df = imdb_df.sample(frac=0.2, random_state=999)


# Data Cleaning
df_train_nan = census_train.replace('?', np.NaN)
df_test_nan  = census_test.replace( '?', np.NaN)
