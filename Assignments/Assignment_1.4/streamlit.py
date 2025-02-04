import streamlit as st
from joblib import dump, load

# import tensorflow as tf
from tensorflow.keras.models import load_model

# import pickle
import spacy
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from gensim.models import KeyedVectors
import numpy as np

# impport hf access token


# hf_access_token = env("HF_ACCESS_TOKEN")
# login(hf_access_token)

# 1. Title and Description
# - Display a title for the app

st.title("Genre Classification")

# - Add a brief description explaining what the app does

st.write("This app classifies text into different genres using a pre-trained model.")


# 2. Model and Preprocessing Setup
# - Load a pre-trained classification model
model = load_model("my_model.h5")
# Load the model from pickle file
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# - Load a text vectorizer (e.g., TF-IDF or CountVectorizer)
repo_id = "NathaNn1111/word2vec-google-news-negative-300-bin"
filename = "GoogleNews-vectors-negative300.bin"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)


# - Load a language processing tool (e.g., SpaCy for tokenization, lemmatization, and stopword removal)
nlp = spacy.load("en_core_web_lg")


# 3. Preprocessing Function
# - Define a function to clean and preprocess the input text by removing stopwords, punctuation, and applying lemmatization
def predict_genre(description, model, word2vec, nlp):
    # Tokenize and create a mean vector for the description
    tokens = [token.text.lower() for token in nlp(description) if token.is_alpha]
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
    else:
        mean_vector = np.zeros(word2vec.vector_size)

    # - Predict the category using the pre-trained model
    prediction = model.predict(mean_vector.reshape(1, -1))[0][0]

    # - Map the predicted output to a readable label
    genre = "horror" if prediction > 0.5 else "romance"
    confidence = prediction if genre == "horror" else 1 - prediction
    return genre, confidence


# 4. User Input
# - Add a st.text_area for users to input the text they want to classify
description = st.text_input("Write a movie description")

# 5. Prediction and Output

# - On button click, preprocess the input text

if st.button("Predict genre"):
    print("button clicked....")
    if not description:
        st.error("Please enter a description.")
    else:
        # - Transform the text using the vectorizer
        try:
            genre, confidence = predict_genre(description, model, word2vec, nlp)
        except Exception as e:
            print("Error :", e)

        st.success(f"Predicted genre: {genre} (Confidence: {confidence:.2f})")
        st.write(
            f"Description: {description}\nPredicted genre: {genre} (Confidence: {confidence:.2f})\n"
        )


# - Display the predicted label

# 6. Error Handling
# - Ensure there is appropriate feedback if the user input is empty or invalid
