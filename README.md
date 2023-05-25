Project Name :
Nepali Number Plate Detection and Character Recognition using Haar Cascade Classifer and Support Vector Machine (SVM)

Project Overview :
The purpose of this project is to develop a Nepali number plate recognition system using a combination of Haar cascade classifier and Support Vector Machine (SVM) algorithm. The goal is to accurately detect and recognize the characters on Nepali number plates from input images.

The project involves two main components: the Haar cascade classifier and the SVM model. The Haar cascade classifier is used for object detection to identify the number plate regions in the input images. It utilizes a set of pre-trained features and a machine learning algorithm to detect objects with specific patterns, in this case, the number plates.

Once the number plate regions are detected, the SVM model is employed for character recognition. The model is trained using a labeled dataset of Nepali characters. The images of characters are preprocessed, resized, and converted to grayscale. The flattened feature vectors of these character images are used to train the SVM model. During recognition, the detected characters from the number plate region are preprocessed similarly and fed into the trained SVM model to predict the corresponding characters.

The overall goal is to achieve accurate and reliable recognition of Nepali number plates, enabling applications such as automated toll collection, parking management, and traffic surveillance. The project combines the strengths of Haar cascade classifier for object detection and SVM for character recognition to accomplish this objective.

