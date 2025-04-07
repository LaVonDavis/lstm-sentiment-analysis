from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

def build_lstm_model(max_words=20000, embedding_dim=300, max_sequence_length=600):
    """Build LSTM model architecture"""
    inputs = Input(shape=(max_sequence_length,))
    x = Embedding(max_words, embedding_dim, input_length=max_sequence_length)(inputs)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
