import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class EyeStateDataLoader:
    def __init__(self, folder_paths, sequence_length):
        self.folder_paths = folder_paths
        self.sequence_length = sequence_length

    @staticmethod
    def extract_labels_from_filename(filename):
        # Extract only the eye_state feature
        eye_state = int(filename[16])
        return eye_state

    def create_sequences(self, X, y):
        # Create sequences of images and labels
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])  # Use the label of the last image in the sequence
        return np.array(X_seq), np.array(y_seq)

    def load_data_from_folder(self, folder_path):
        X, y = [], []
        for filename in sorted(os.listdir(folder_path)):
            if os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith('.png'):
                img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (32, 32))  # Resize images
                img = img / 255.0  # Normalize pixel values

                label = self.extract_labels_from_filename(filename)

                X.append(img)
                y.append(label)
        return X, y

    def load_data(self):
        all_X, all_y = [], []

        for folder_path in self.folder_paths:
            X, y = self.load_data_from_folder(folder_path)
            all_X.extend(X)
            all_y.extend(y)

        X_seq, y_seq = self.create_sequences(all_X, all_y)

        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
