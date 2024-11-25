import string

import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import sent_tokenize

class CleanUp:
    def __init__(self, stopwords_lan="english"):
        # Download necessary NLTK resources
        nltk.download("stopwords")
        nltk.download('punkt_tab')
        nltk.download("wordnet")
        self.stop_words = set(stopwords.words(stopwords_lan))
        # Lemmatization
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        return " ".join([word for word in words if word not in self.stop_words])

    def lemmatize_text(self, text):
        words = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])

    def token_sentence(self,text):

        sentences = sent_tokenize(text)
        return sentences
        # Output: ['This is the first sentence.', 'Here is another one.', 'This is the third.']

    def cleanup(self, df: pandas.DataFrame,dataTag):
        # Lowercase
        df[dataTag] = df[dataTag].str.lower()

        # Remove punctuation
        # df[dataTag] = df[dataTag].str.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers
        # df[dataTag] = df[dataTag].str.replace(r"\d+", "numbers", regex=True)

        # Remove URLs
        df[dataTag] = df[dataTag].str.replace(r"http\S+|www\S+", "urls", regex=True)

        # Remove extra whitespace
        df[dataTag] = df[dataTag].str.strip()
        df[dataTag] = df[dataTag].apply(self.remove_stopwords)
        df[dataTag] = df[dataTag].apply(self.lemmatize_text)
        df[dataTag] = df[dataTag].apply(self.token_sentence)

        return df
