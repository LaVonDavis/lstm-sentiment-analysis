import matplotlib.pyplot as plt
from .model import build_lstm_model
from .data_loading import load_imdb_data
from .config import Config

def train_model():
    """Train LSTM model and save training history"""
    (x_train, y_train), (x_test, y_test) = load_imdb_data(
        Config.MAX_NB_WORDS,
        Config.MAX_SEQUENCE_LENGTH
    )
    
    model = build_lstm_model(
        Config.MAX_NB_WORDS,
        Config.EMBEDDING_DIM,
        Config.MAX_SEQUENCE_LENGTH
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(x_test, y_test)
    )
    
    return history, model
