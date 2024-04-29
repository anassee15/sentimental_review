import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import string
import os




# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text


def clean_text(text):
    return text.strip().lower()


class TfidfPipepline:
    def __init__(self, n_train=40000, pkl_name=""):

        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = list(STOP_WORDS)
        self.punctuations = string.punctuation

        if pkl_name == "":
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../data/imdb_shuffle.csv')
            pkl_name = "tfidf.pkl"
            df = pd.read_csv(filename)
            df = df[:n_train]
            # Vectorization
            vectorizer = CountVectorizer(
                tokenizer=self.spacy_tokenizer, ngram_range=(1, 1))
            classifier = LinearSVC()

            # Using Tfidf
            tfvectorizer = TfidfVectorizer(tokenizer=self.spacy_tokenizer)

            X = df['Review']
            ylabels = df['Sentiment']

            # Features and Labels

            X_train, X_test, y_train, y_test = train_test_split(
                X, ylabels, test_size=0.2, random_state=0)

            # Create the  pipeline to clean, tokenize, vectorize, and classify
            pipe = Pipeline([("cleaner", predictors()),
                            ('vectorizer', tfvectorizer),
                            ('classifier', classifier)])

            # Fit our data
            pipe.fit(X_train, y_train)

            pickle.dump(pipe, open(pkl_name, "wb"))

        self._train_model = pickle.load(open(pkl_name, 'rb'))
        

    def spacy_tokenizer(self, sentence):
        mytokens = self.nlp(sentence)
        mytokens = [word.lemma_.lower().strip() if word.lemma_ !=
                    "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [
            word for word in mytokens if word not in self.stopwords and word not in self.punctuations]
        return mytokens


class CountVectorizerPipepline:
    def __init__(self, n_train=40000, pkl_name=""):

        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = list(STOP_WORDS)
        self.punctuations = string.punctuation

        if pkl_name == "":
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../data/imdb_shuffle.csv')
            pkl_name = "countVectorizer.pkl"
            df = pd.read_csv(filename)
            df = df[:n_train]
            # Vectorization
            vectorizer = CountVectorizer(
                tokenizer=self.spacy_tokenizer, ngram_range=(1, 1))
            classifier = LinearSVC()

            X = df['Review']
            ylabels = df['Sentiment']

            # Features and Labels

            X_train, X_test, y_train, y_test = train_test_split(
                X, ylabels, test_size=0.2, random_state=0)

            # Create the  pipeline to clean, tokenize, vectorize, and classify
            pipe = Pipeline([("cleaner", predictors()),
                            ('vectorizer', vectorizer),
                            ('classifier', classifier)])

            # Fit our data
            pipe.fit(X_train, y_train)

            pickle.dump(pipe, open(pkl_name, "wb"))  
        self._train_model = pickle.load(open(pkl_name, 'rb'))

    def spacy_tokenizer(self, sentence):
        mytokens = self.nlp(sentence)
        mytokens = [word.lemma_.lower().strip() if word.lemma_ !=
                    "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [
            word for word in mytokens if word not in self.stopwords and word not in self.punctuations]
        return mytokens
