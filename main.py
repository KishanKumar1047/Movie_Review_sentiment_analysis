# Step 1: Import Libraries and Build Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Rebuild the original model architecture
model = Sequential([
    Input(shape=(500,), name='input_layer'),
    Embedding(input_dim=10000, output_dim=32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

# Load the pre-trained weights
model.load_weights('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App
import streamlit as st

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as **positive** or **negative**.')

# User input
user_input = st.text_area('✍️ Movie Review')

if st.button('🔍 Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'

    # Display the result
    st.subheader(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: `{prediction[0][0]:.4f}`')
else:
    st.info('Please enter a movie review above and click Classify.')
