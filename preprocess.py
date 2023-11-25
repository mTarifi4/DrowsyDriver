import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    return image

def create_sequences(images, labels, time_steps):
    X, y = [], []
    for i in range(len(images) - time_steps):
        X.append(images[i:i + time_steps])
        y.append(labels[i + time_steps - 1])  # Label for the last image in the sequence
    return np.array(X), np.array(y)

# Example file paths and labels
file_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
labels = [0, 1, ...]  # 0 for not sleeping, 1 for sleeping

# Load and preprocess images
processed_images = [load_and_preprocess_image(path) for path in file_paths]

# Create sequences
time_steps = 5
X, y = create_sequences(processed_images, labels, time_steps)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
