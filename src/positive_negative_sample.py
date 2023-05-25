import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# path to directory containing positive image and xml files
positive_dir = './data/positive/'

# path to directory containing negative images
negative_dir = './data/negative/'

# path to directory where positive and negative samples will be stored
output_dir = './data/sample/'

# size of positive and negative samples
sample_size = (50, 50)

# positive sample
with open('./data/positive.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    image_path, xml_path = line.strip().split()
    # print(image_path, xml_path)
    image_name = os.path.basename(image_path)
    xml_name = os.path.basename(xml_path)
    # print(image_name, xml_name)
    image = cv2.imread(os.path.join(positive_dir, image_name))
    
    # use xml tree to extract bounding box coordinates
    tree = ET.parse(os.path.join(positive_dir, xml_name))
    root = tree.getroot()
    xmin = int(root.find('object').find('bndbox').find('xmin').text)
    ymin = int(root.find('object').find('bndbox').find('ymin').text)
    xmax = int(root.find('object').find('bndbox').find('xmax').text)
    ymax = int(root.find('object').find('bndbox').find('ymax').text)
    
    license_plate = image[ymin:ymax, xmin:xmax]
    license_plate = cv2.resize(license_plate, sample_size)
    output_path = os.path.join(output_dir, f'positive_{xmin}_{ymin}.jpg')
    cv2.imwrite(output_path, license_plate)

# negative sample
for filename in os.listdir(negative_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(negative_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, sample_size)
        output_path = os.path.join(output_dir, f'negative_{filename}')
        cv2.imwrite(output_path, image)