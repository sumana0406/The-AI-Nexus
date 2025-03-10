import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = Perceptron()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


new_email = ["Meeting at 10 tomorrow!"]
new_email_tfidf = vectorizer.transform(new_email)
prediction = model.predict(new_email_tfidf)
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")

new_email1 = ["Congrats! You've won a free iPhone!"]
new_email_tfidf1 = vectorizer.transform(new_email1)
prediction1 = model.predict(new_email_tfidf1)
print("Prediction:", "Spam" if prediction1[0] == 1 else "Not Spam")

