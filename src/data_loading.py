import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_imdb_data(max_words=20000, max_sequence_length=600):
    """Load and preprocess IMDB dataset"""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    
    x_train = pad_sequences(x_train, maxlen=max_sequence_length)
    x_test = pad_sequences(x_test, maxlen=max_sequence_length)
    
    return (x_train, y_train), (x_test, y_test)
