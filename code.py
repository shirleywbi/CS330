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

X = df[['lon', 'lat']] # TODO: Use the drop example
y = df['vote']

# Exploratory
df.describe()
X.shape
X.ndim
y.value_counts()

## Scatterplot
scatter = plt.scatter(df["meat"], df["grade"], c=df["cilantro"]=="Yes", cmap=plt.cm.coolwarm);
plt.xlabel("Meat consumption (% days)")
plt.ylabel("Expected grade (%)")
plt.legend(scatter.legend_elements()[0], ["No", "Yes"])

scatter.legend_elements()[0]

# Splitting the data
df_train, df_test = train_test_split(df)