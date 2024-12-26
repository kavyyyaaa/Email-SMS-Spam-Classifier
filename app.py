import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    ps = PorterStemmer()

    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [i for i in text if i.isalnum()]

    stop_words = set(stopwords.words('english'))

    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    text = [ps.stem(i) for i in text]

    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier ")

input_sms = st.text_input("Enter the Message")
# 1 preprocess
transformed_sms = transform_text(input_sms)
# 2 vectorize
vector_input = tfidf.transform([transformed_sms])
# 3 predict
result = model.predict(vector_input)[0]
# 4 display
if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")
