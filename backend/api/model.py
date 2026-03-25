import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Model başlıyor...")

# dataset yükle
data = pd.read_csv("C:/Users/luban/smart-feedback-analyzer/dataset/IMDB Dataset.csv")

# sütunlar
texts = data["review"]
labels = data["sentiment"]

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# vectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# tahmin
y_pred = model.predict(X_test_vec)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Model hazır 🚀")

# predict fonksiyonu
def predict_sentiment(text):
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]