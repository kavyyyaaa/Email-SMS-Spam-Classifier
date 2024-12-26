import os
import pickle
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download the NLTK punkt resource if not already downloaded
nltk.download('punkt', quiet=True)

# Set the directory to save model and vectorizer
save_dir = os.path.expanduser("~/sms_spam_detection")
os.makedirs(save_dir, exist_ok=True)

# Define paths to store model and vectorizer
model_path = os.path.join(save_dir, "model.pkl")
vectorizer_path = os.path.join(save_dir, "vectorizer.pkl")

# Train the model if it does not exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    training_data = ["Free money now!", "Hey, how are you?", "Limited offer, click now!", "Let's catch up soon."]
    labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

    # Initialize vectorizer and transform training data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, labels)

    # Save the trained model and vectorizer
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    st.success("Model and vectorizer trained and saved successfully.")
else:
    st.info("Model and vectorizer already exist.")

# Load the trained model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to transform the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    return " ".join(text)  # Join tokens back into a single string

# Streamlit App
st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the Message")

if st.button("Predict"):
    if input_sms:
        # Transform the input message
        transformed_sms = transform_text(input_sms)
        # Convert the transformed message into a vector using the same vectorizer
        vector_input = vectorizer.transform([transformed_sms])

        # Check if the number of features matches between the model and input data
        if vector_input.shape[1] != model.n_features_in_:
            st.error("Feature mismatch: Please ensure the vectorizer and model are aligned.")
        else:
            # Make the prediction
            result = model.predict(vector_input)[0]
            st.header("Spam Message" if result == 1 else "Not a Spam Message")
