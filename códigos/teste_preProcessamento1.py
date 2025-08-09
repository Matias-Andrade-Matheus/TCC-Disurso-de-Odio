import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
     
import re
import string
import unicodedata

lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download("wordnet")
stop_words = stopwords.words('portuguese')

def clean_text(text):
    '''
    Perform stop-words removal and lemmatization
    '''
       
    text = text.lower()

    text_normalize = unicodedata.normalize("NFD", text)
    text = ''.join(
        char for char in text_normalize
        if not unicodedata.combining(char)
    )
            
    text = re.sub(r"[^a-zA-Z?.!,¿]+|http\S+", " ", text)

    not_punctuation_text = ""
    for char in text:
        if char not in string.punctuation:
            not_punctuation_text += char
        else:
            not_punctuation_text += " "
    text = not_punctuation_text

    words = [word for word in text.split() if word not in stopwords.words('portuguese')]
    words = [WordNetLemmatizer().lemmatize(word) for word in words]

    return " ".join(words)
