import os

# directory to positive and negative images
positive_dir = './data/positive'
negative_dir = './data/negative'

# output file for positive and negative images
positive_file = './data/positive.txt'
negative_file = './data/negative.txt'

# generate positive file
with open(positive_file, 'w') as f:
    for root, dirs, files in os.walk(positive_dir):
        for file in files:
            if file.endswith('.jpg'):
                xml_file = os.path.splitext(file)[0] + '.xml'
                xml_path = os.path.join(root, xml_file)
                image_path = os.path.join(root, file)
                f.write(image_path + ' ' + xml_path + '\n')

# generate negative file
with open(negative_file, 'w') as f:
    for root, dirs, files in os.walk(negative_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                f.write(image_path + '\n')