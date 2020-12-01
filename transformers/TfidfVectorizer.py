from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
            "This is the first document, the FIRST",
            "This is the second document",
            "This is the third document",
            "This is the fourth document"
         ]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
header = vectorizer.get_feature_names()
labels = ['D1', 'D2', 'D3', 'D4']
df = pd.DataFrame(X.toarray(), columns = header, index = labels)