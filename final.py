import os
import pickle
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download 'punkt' explicitly if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.write("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Create a directory for saving model and vectorizer
save_dir = os.path.expanduser("~/sms_spam_detection")
os.makedirs(save_dir, exist_ok=True)

# Define paths for model and vectorizer
model_path = os.path.join(save_dir, "model.pkl")
vectorizer_path = os.path.join(save_dir, "vectorizer.pkl")

# Train and save the model if not already saved
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    # Sample training data and labels
    training_data = ["Free money now!", "Hey, how are you?", "Limited offer, click now!", "Let's catch up soon."]
    labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

    # Vectorize the training data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, labels)

    # Save the model and vectorizer to disk
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer trained and saved successfully.")
else:
    print("Model and vectorizer already exist.")

# Load the model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess and tokenize the text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize using punkt
    return " ".join(text)

# Streamlit app interface
st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the Message")

if st.button("Predict"):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])

        # Check for feature mismatch
        if vector_input.shape[1] != model.n_features_in_:
            st.error("Feature mismatch: Please ensure the vectorizer and model are aligned.")
        else:
            result = model.predict(vector_input)[0]
            # Display the prediction result
            st.header("Spam Message" if result == 1 else "Not a Spam Message")
    else:
        st.warning("Please enter a message to classify.")




