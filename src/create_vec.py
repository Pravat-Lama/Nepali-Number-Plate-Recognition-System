import cv2
import numpy as np
import os

# path to the text file containing paths to the positive samples
# positive_samples_file = './annotations.txt'
positive_samples_file = './data/positive.txt'

# path to the directory containing negative samples
negative_samples_dir = './data/negative/'

# output path for the .vec file
output_file = 'samples.vec'

# size of the sample images
sample_size = (50, 50)

# read the paths of positive samples from the text file
with open(positive_samples_file, 'r') as f:
    positive_samples_paths = f.readlines()

# remove any leading or trailing whitespace characters from the paths
# positive_samples_paths = [path.strip() for path in positive_samples_paths]

# initialize an empty list to store the positive samples
positive_samples = []

# read and resize each positive sample image and append it to the list
for path in positive_samples_paths:
    image_path, xml_path = path.strip().split()
    # print(image_path)
    image_name = os.path.basename(image_path)
    # print(image_name)
    img = cv2.imread(os.path.join('./data/positive/', image_name))
    img_resized = cv2.resize(img, sample_size)
    positive_samples.append(img_resized)

# create an array to store the positive samples
positive_samples_array = np.array(positive_samples)

# create an array to store the labels for the positive samples (1 for positive)
positive_labels = np.ones(positive_samples_array.shape[0], np.int32)

# create an array to store the negative samples
negative_samples = []

# read and resize each negative sample image and append it to the list
for filename in os.listdir(negative_samples_dir):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(negative_samples_dir, filename))
        img_resized = cv2.resize(img, sample_size)
        negative_samples.append(img_resized)

# create an array to store the negative samples
negative_samples_array = np.array(negative_samples)

# create an array to store the labels for the negative samples (0 for negative)
negative_labels = np.zeros(negative_samples_array.shape[0], np.int32)

# concatenate the positive and negative samples and labels
samples_array = np.concatenate((positive_samples_array, negative_samples_array), axis=0)
labels = np.concatenate((positive_labels, negative_labels), axis=0)

# create the .vec file using the cv2.imwrite function
with open(output_file, 'wb') as f:
    f.write(np.array([0, 0, 0, 0], np.int32).tobytes())
    f.write(np.array([len(samples_array)], np.int32).tobytes())
    for i in range(len(samples_array)):
        img = samples_array[i]
        img_bytes = img.tobytes()
        label_bytes = np.array([labels[i]], np.int32).tobytes()
        f.write(label_bytes)
        f.write(np.array([sample_size[1], sample_size[0]], np.int32).tobytes())
        f.write(img_bytes)
