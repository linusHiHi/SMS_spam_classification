import string

import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


class CleanUp:
    def __init__(self, df, stopwords_lan="english"):
        # Download necessary NLTK resources
        nltk.download("stopwords")
        nltk.download('punkt_tab')
        nltk.download("wordnet")
        self.df = df
        self.stop_words = set(stopwords.words(stopwords_lan))
        # Lemmatization
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        return " ".join([word for word in words if word not in self.stop_words])

    def lemmatize_text(self, text):
        words = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])

    def cleanup(self, df: pandas.DataFrame):
        # Lowercase
        df["Message"] = df["Message"].str.lower()

        # Remove punctuation
        df["Message"] = df["Message"].str.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers
        # df["Message"] = df["Message"].str.replace(r"\d+", "numbers", regex=True)

        # Remove URLs
        df["Message"] = df["Message"].str.replace(r"http\S+|www\S+", "urls", regex=True)

        # Remove extra whitespace
        df["Message"] = df["Message"].str.strip()
        df["Message"] = df["Message"].apply(self.remove_stopwords)
        df["Message"] = df["Message"].apply(self.lemmatize_text)
        return df
