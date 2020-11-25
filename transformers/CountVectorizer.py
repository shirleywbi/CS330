from sklearn.feature_extraction.text import CountVectorizer

imdb_df = pd.read_csv('data/imdb_master.csv', index_col=0, encoding="ISO-8859-1")
imdb_df = imdb_df[imdb_df['label'].str.startswith(('pos','neg'))]
imdb_df = imdb_df.sample(frac=0.2, random_state=999)

imdb_train, imdb_test = train_test_split(imdb_df, random_state=123)

X_train_imdb_raw = imdb_train['review']
y_train_imdb = imdb_train['label']

X_test_imdb_raw = imdb_test['review']
y_test_imdb = imdb_test['label']

vec = CountVectorizer(min_df=50, binary=True) # present/absent
X_train_imdb = vec.fit(X_train_imdb_raw)

# WARNING: Do NOT fit the transformer with the test data
# vec.fit(X_test_imdb_raw);

X_test_imdb = vec.transform(X_test_imdb_raw)

dt = DecisionTreeClassifier()
dt.fit(X_train_imdb, y_train_imdb)

dt.score(X_train_imdb, y_train_imdb)
dt.score(X_test_imdb, y_test_imdb)

# Getting feature names
vec.get_feature_names()