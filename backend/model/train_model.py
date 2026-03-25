from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this",
    "I hate this"
]

labels = ["positive", "negative"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

test = ["I love it"]
X_test = vectorizer.transform(test)

prediction = model.predict(X_test)

print(prediction[0])