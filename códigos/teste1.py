import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
     
import re
import string

# df = pd.read_csv('/content/UpdatedResumeDataSet.csv')
# df.head()

stop_words = stopwords.words('portuguese')

def clean_text(text):
    '''
    Perform stop-words removal and lemmatization
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+|http\S+", " ", text)
    text = ''.join([char for char in text if char not in string.punctuation])
    words = [word for word in text.split() if word not in stopwords.words('portuguese')]
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    print(words)
    return " ".join(words)

# texto_limpo = clean_text("o rato roeu a roupa do rei da Roma.")

texto_limpo = clean_text("três mafagafos pariram três mafagafinhos em um ninho de mafagafos")

print(texto_limpo)