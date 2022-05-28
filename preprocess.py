import re
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def clean_teks(teks):
    teks_clean = re.sub(r"\d+", " ", teks) 
    teks_clean = teks_clean.lower()
    teks_clean = re.sub(r"[^\w\s]", " ", teks_clean)
    teks_clean = re.sub(r"\s+", " ", teks_clean)
    teks_clean = teks_clean.strip()
    return teks_clean

def remove_stopwords(teks):
    cleaned = [word for word in nltk.word_tokenize(teks) if word not in stopwords]
    return " ".join(cleaned)

def lemmatization(teks):
    lemma = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(teks)]
    return " ".join(lemma)

def pre_process(teks):
    teks_clean = clean_teks(teks)
    teks_clean = remove_stopwords(teks_clean)
    teks_clean = lemmatization(teks_clean)
    return teks_clean