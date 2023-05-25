import cv2
import os
import numpy as np

# Path to directory containing positive and negative samples
samples_dir = './data/sample/'

# Size of samples
sample_size = (50, 50)

# Load positive and negative samples
pos_samples = []
neg_samples = []
for filename in os.listdir(samples_dir):
    if filename.startswith('positive'):
        image = cv2.imread(os.path.join(samples_dir, filename))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pos_samples.append(image_gray)
    elif filename.startswith('negative'):
        image = cv2.imread(os.path.join(samples_dir, filename))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        neg_samples.append(image_gray)

# Create list of labels for samples
labels = [1] * len(pos_samples) + [0] * len(neg_samples)

# Create list of all samples
samples = pos_samples + neg_samples

# Convert samples to numpy array and convert data type to CV_32F
samples = np.array(samples, dtype=np.float32)

# Reshape samples to 2D array
samples = samples.reshape(len(samples), -1)

# Convert labels to numpy array
labels = np.array(labels)

# Train Haar cascade classifier
cascade_params = {'minSize': (50, 50), 'maxSize': (200, 200)}
cascade = cv2.CascadeClassifier()
cascade.train(samples, labels, cascade_params)
cascade.save('./models/my_classifier.xml')
