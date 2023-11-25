from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.applications import VGG16

# Define CNN model for feature extraction
def create_cnn():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    return model

# Define full model
def create_model(time_steps, cnn_model):
    model = Sequential()
    model.add(TimeDistributed(cnn_model, input_shape=(time_steps, 224, 224, 3)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Time steps (number of sequential images to consider)
time_steps = 5

# Create and compile the model
cnn_model = create_cnn()
model = create_model(time_steps, cnn_model)
model.summary()
