import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


model = joblib.load('models.joblib')
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model_spam = pickle.load(open('model.pkl', 'rb'))

st.title("scaning web page what contents spam or terrorism spreading site")

input_url = st.text_input("Enter the URL")


def getdata(t):
    r = requests.get(t)
    return r.text


if st.button('scan'):
    
    htmldata = getdata(input_url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    data = ''
    h1 = []
    p = []
    
    for data in soup.find_all("h1"):
        h1.append(data.get_text())
    for data in soup.find_all("p"):
        p.append(data.get_text())

    predict_count = 0
    terror_detect = 0
    predict_count_spam = 0
    spam_detect = 0

    for item in p:
        transformed_sms = transform_text(item)
    # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
    # 3. predict
        model_spam.predict(vector_input)[0]
        predict_count_spam += 1

        if model_spam.predict(vector_input)[0] == 1:
            spam_detect += 1

    for item in h1:
        transformed_sms = transform_text(item)
    # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
    # 3. predict
        model_spam.predict(vector_input)[0]
        predict_count_spam += 1

        if model_spam.predict(vector_input)[0] == 1:
            spam_detect += 1

    for item in p:
        model.predict(np.expand_dims(item, 0))
        predict_count += 1

        if model.predict(np.expand_dims(item, 0))[0] == 1:
            terror_detect += 1

    for i in h1:
        model.predict(np.expand_dims(i, 0))
        predict_count += 1

        if model.predict(np.expand_dims(i, 0))[0] == 1:
            terror_detect += 1

    perc = (terror_detect/predict_count) * 100
    res_ter = round(perc)
    percentage = (spam_detect/predict_count_spam) * 100
    res_spam = round(percentage)
    st.header("Spam : %  \n")
    st.header("terrorism spreading: "+str(res_ter)+"% ")

