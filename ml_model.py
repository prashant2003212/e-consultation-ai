
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

def load_training_data():
    # small example dataset — replace/extend with real labeled data later
    data = {
        "text": [
            "I support this law completely",
            "This amendment is beneficial",
            "Very helpful reform",
            "I am against this rule",
            "This proposal is bad",
            "Worst policy ever",
            "Good step by the government",
            "This rule will create problems",
            "Excellent draft",
            "I disagree with this reform",
            "Not good, will harm public",
            "Great initiative, well done",
            "I strongly oppose this",
            "Positive step forward",
            "Negative impact expected"
        ],
        "label": [
            "positive","positive","positive",
            "negative","negative","negative",
            "positive","negative","positive",
            "negative","negative","positive",
            "negative","positive","negative"
        ]
    }
    return pd.DataFrame(data)

def train_model(verbose=False):
    df = load_training_data()
    X, y = df['text'], df['label']
    # simple ensemble of LR, SVM, NB
    lr = LogisticRegression(max_iter=400)
    svm = LinearSVC(max_iter=400)
    nb = MultinomialNB()
    ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm), ('nb', nb)], voting='hard')
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')),
        ('clf', ensemble)
    ])
    model.fit(X, y)
    if verbose:
        print("Trained new ML ensemble model with sample data.")
    return model

# convenience wrappers used by app
_MODEL = None

def ensure_model_loaded():
    global _MODEL
    if _MODEL is None:
        _MODEL = train_model()
    return _MODEL

def analyze_comment(text):
    model = ensure_model_loaded()
    return model.predict([text])[0]

def analyze_batch(list_of_texts):
    model = ensure_model_loaded()
    return model.predict(list_of_texts)
