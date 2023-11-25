import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

FOLDER_PATH = "/Users/omama/Desktop/mrlEyes_2018_01/s0001"
SEQUENCE_LENGTH = 5  # Length of the image sequences

def extract_labels_from_filename(filename):
    # Extract only the eye_state feature
    eye_state = int(filename[16])
    return eye_state

def create_sequences(X, y, sequence_length):
    # Create sequences of images and labels
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])  # Use the label of the last image in the sequence
    return np.array(X_seq), np.array(y_seq)

# Lists to store image data and labels
X, y = [], []

# Iterate through all files in the folder and preprocess images
for filename in sorted(os.listdir(FOLDER_PATH)):
    if os.path.isfile(os.path.join(FOLDER_PATH, filename)) and filename.endswith('.png'):
        img = cv2.imread(os.path.join(FOLDER_PATH, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))  # Resize images
        img = img / 255.0  # Normalize pixel values

        # Extract the eye_state label
        label = extract_labels_from_filename(filename)

        # Append data and label to the lists
        X.append(img)
        y.append(label)

# Create sequences of data
X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

print("Number of Sequences:", len(X_seq))
