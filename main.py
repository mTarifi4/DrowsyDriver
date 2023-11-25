base_path = "/Users/omama/Desktop/mrlEyes_2018_01/s000"
folder_paths = [f"{base_path}{i:02d}" for i in range(1, 38)]  # Generates paths from s0001 to s0037
SEQUENCE_LENGTH = 5

data_loader = EyeStateDataLoader(folder_paths, SEQUENCE_LENGTH)
X_train, X_test, y_train, y_test = data_loader.load_data()

print("Number of Training Sequences:", len(X_train))
print("Number of Testing Sequences:", len(X_test))
