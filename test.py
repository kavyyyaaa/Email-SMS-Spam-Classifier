import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Text preprocessing function
def transform_text(text):
    ps = PorterStemmer()

    text = text.lower()
    text = nltk.word_tokenize(text)  # Tokenize the text
    text = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    text = [i for i in text if i not in stop_words and i not in string.punctuation]  # Remove stopwords and punctuation
    text = [ps.stem(i) for i in text]  # Apply stemming
    return " ".join(text)


# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except EOFError as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Streamlit app
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the Message")

# Process the input if entered
if input_sms:
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the processed SMS
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict whether the SMS is spam or not
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
else:
    st.warning("Please enter a message to classify.")
