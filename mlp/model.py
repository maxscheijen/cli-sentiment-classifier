import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class Model:
    def __init__(self, vocab_size=10_000):
        self.vocab_size = vocab_size
        self.clf = None
        self.vectorizer = None

    def train(self, X_train, y_train):
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        X_train = self.vectorizer.fit_transform(X_train)

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        X = self.vectorizer.transform(X)

        self.y_pred = self.clf.predict(X)
        return self.y_pred

    def predict_proba(self, X):
        X = self.vectorizer.transform(X)

        self.y_proba = self.clf.predict_proba(X)
        return self.y_proba

    def save_model(self):
        joblib.dump(self.vocab_size, "models/vocab.pkl")
        joblib.dump(self.vectorizer, "models/vecorizer.pkl")
        joblib.dump(self.clf, "models/model.pkl")

    @staticmethod
    def load_model():
        model = Model()
        model.vocab_size = joblib.load("models/vocab.pkl")
        model.vectorizer = joblib.load("models/vecorizer.pkl")
        model.clf = joblib.load("models/model.pkl")
        return model
