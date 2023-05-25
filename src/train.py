import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Function to preprocess an image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to a fixed size (e.g., 32x32)
    resized_image = cv2.resize(image, (32, 32))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to a range of 0-1
    normalized_image = gray_image.astype(np.float32) / 255.0

    return normalized_image

# Set the path to your labeled dataset
dataset_path = './data/characters'

# List to store image paths and corresponding labels
image_paths = []
labels = []

# Iterate through each character class folder in the dataset
for character_class in os.listdir(dataset_path):
    character_class_path = os.path.join(dataset_path, character_class)
    
    # Iterate through each image file in the character class folder
    for image_file in os.listdir(character_class_path):
        image_path = os.path.join(character_class_path, image_file)
        
        # Add the image path and label to the lists
        image_paths.append(image_path)
        labels.append(character_class)

# Shuffle the dataset
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)

# Preprocess the training images
X_train = np.array([preprocess_image(image) for image in train_images])

# Flatten the training images
X_train_flat = X_train.reshape(len(X_train), -1)

# Preprocess the testing images
X_test = np.array([preprocess_image(image) for image in test_images])

# Flatten the testing images
X_test_flat = X_test.reshape(len(X_test), -1)

# Create an instance of the classifier
model = SVC()

# Train the model
model.fit(X_train_flat, train_labels)

# Evaluate the model on the testing data
accuracy = model.score(X_test_flat, test_labels)
print(f"Accuracy on testing data: {accuracy}")

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
