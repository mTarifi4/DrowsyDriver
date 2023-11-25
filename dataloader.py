import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


FOLDER_PATH = "/Users/omama/Desktop/mrlEyes_2018_01/s0001"

def extract_labels_from_filename(filename):
    subject_id = int(filename[1:5])
    image_id = int(filename[6:11])
    gender = int(filename[12])
    glasses = int(filename[14])
    eye_state = int(filename[16])
    reflections = int(filename[18])
    lighting_conditions = int(filename[20])
    sensor_id = int(filename[22:24])

    return subject_id, image_id, gender, glasses, eye_state, reflections, lighting_conditions, sensor_id




# Iterate through all files in the folder
for filename in os.listdir(FOLDER_PATH):

    if os.path.isfile(os.path.join(FOLDER_PATH, filename)) and filename.endswith('.png'):
        labels = extract_labels_from_filename(filename)
        print(f"File: {filename}, Labels: {labels}")

#eg File: s0001_01343_0_1_0_2_0_01.png, Labels: (1, 1343, 0, 1, 0, 2, 0, 1)


# Lists to store image data (X) and labels (y)
X = []
y = []

# Iterate through all files in the folder
for filename in os.listdir(FOLDER_PATH):
    if os.path.isfile(os.path.join(FOLDER_PATH, filename)) and filename.endswith('.png'):
        # Load and preprocess the image
        #img = cv2.imread(os.path.join(FOLDER_PATH, filename))
        img = cv2.imread(os.path.join(FOLDER_PATH, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))  # Adjust dimensions as needed
        img = img / 255.0  # Normalize pixel values to be between 0 and 1

        # Extract labels
        labels = extract_labels_from_filename(filename)

        # Append data and labels to the lists
        X.append(img)
        y.append(labels)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X)
