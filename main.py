import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
#GenAI-Simple RNN-Classification-Movie-Review
## Mapping of words index back to words (for our understanding)
word_index = imdb.get_word_index()
# word_index
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

## Helper funcitons
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



## Streamlit App
import streamlit as st

st.title('IMDB Movie Revie Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

## User Input
user_input = st.text_area('Movie Review')
sample_review = "This is a test review. the movie was great"
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    ## Make Prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the result
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction Score : {prediction[0][0]}")
else:
    st.write('Please Enter a Movie Review')
    st.write(f"Sample review : {sample_review}" )
