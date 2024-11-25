import re
import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import sent_tokenize

def rm_punctuation_from_list(df_list):
    return list((re.sub(r'[^\w\s]', '', sent)) for sent in df_list)

def token_sentence(text):

    sentences = sent_tokenize(text)
    return sentences
    # Output: ['This is the first sentence.', 'Here is another one.', 'This is the third.']


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

    def cleanup(self, df: pandas.DataFrame,dataTag):
        # Lowercase
        df[dataTag] = df[dataTag].str.lower()

        # Remove numbers
        # df[dataTag] = df[dataTag].str.replace(r"\d+", "numbers", regex=True)

        # Remove URLs
        df[dataTag] = df[dataTag].str.replace(r"http\S+|www\S+", "urls", regex=True)

        # Remove extra whitespace
        df[dataTag] = df[dataTag].str.strip()
        df[dataTag] = df[dataTag].apply(self.remove_stopwords)
        df[dataTag] = df[dataTag].apply(self.lemmatize_text)
        df[dataTag] = df[dataTag].apply(token_sentence)
        # Remove punctuation
        df[dataTag] = df[dataTag].map(rm_punctuation_from_list)



        return df


