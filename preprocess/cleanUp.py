import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

sourceDataPath = "../dataSet/sms_data.csv"
finalDataPath = "../dataSet/cleaned_sms_Data.csv"

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download("wordnet")

# Load data
df = pd.read_csv(sourceDataPath)

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

# Remove stop words
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

df["Message"] = df["Message"].apply(remove_stopwords)

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])

df["Message"] = df["Message"].apply(lemmatize_text)

# Save cleaned data
df.to_csv(finalDataPath, index=False)
print("finished!")
