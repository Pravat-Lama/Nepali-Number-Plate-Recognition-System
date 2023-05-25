import string
import cv2
from datetime import datetime
import os
import csv
from PIL import Image
import numpy as np
import joblib

# Load the trained classifier
classifier = cv2.CascadeClassifier('./models/haarcascade.xml')

# Load the trained model
model = joblib.load('./models/model.pkl')

def recognize_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform object detection on the test image
    plates = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Create an empty list to store the recognized characters
    texts = []

    for (x, y, w, h) in plates:
        # Extract the detected number plate from the image
        plate = gray[y:y+h, x:x+w]

        # Apply a binary threshold to the number plate to make the characters more visible
        _, binary = cv2.threshold(plate, 150, 255, cv2.THRESH_BINARY)

        # Find contours in the image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set a minimum and maximum area threshold to filter out small or large contours
        min_area = 100
        max_area = 5000

        # List to store the individual character images
        character_images = []

        # Iterate through the contours and extract character images
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            
            # Filter contours based on area
            if min_area < area < max_area:
                # Create a bounding rectangle around the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the character image using the bounding rectangle
                character_image = plate[y:y+h, x:x+w]
                
                # Append the character image to the list
                character_images.append(character_image)
                
                # Draw a rectangle around the detected character on the cropped grayscale image
                cv2.rectangle(plate, (x, y), (x + w, y + h), (255, 255, 255), 2)

        for character_image in character_images:
            # Preprocess the image before feeding it into the model
            normalized_image = character_image.astype(np.float32) / 255.0

            # Flatten the preprocessed image
            flattened_image = normalized_image.reshape(1, -1)
            flattened_image = cv2.resize(flattened_image, (32, 32)).reshape(1, -1)
            predicted_character = model.predict(flattened_image)[0]

            # Store the detected character
            texts.append(str(predicted_character))

    # Get the current date and time
    now = datetime.now()

    # Convert the date and time to string
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

    # Remove any invalid characters from the filename
    filename = f'{"".join(texts)}_{date_time}.jpg'
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)

    # Save the cropped grayscale number plate image to the output directory
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    img = Image.fromarray(plate)
    img.save(path)
    if os.path.exists(path):
        print(f'Grayscale image saved at {path}')
        # Show the cropped grayscale number plate image
        cv2.imshow('Cropped Grayscale Image', plate)
    else:
        print(f'Error: Could not save image at {path}')
        cv2.imshow('Cropped Grayscale Image', plate)

    # Draw a rectangle around the detected number plate on the original image
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the bounding box image
    cv2.imshow('Bounding Box Image', image)

    cv2.waitKey(0)

    # Save the detected text and timestamp to the CSV file
    csv_path = os.path.join(directory, 'results.csv')
    header = ['Text', 'Timestamp']
    rows = [[text, date_time] for text in texts]
    try:
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)
        print(f'Text and timestamp saved to {csv_path}')
    except Exception as e:
        print(f'Error occurred while writing to CSV file: {e}')

    # Return the recognized text and timestamp as a tuple
    print("".join(texts) + "\n" + date_time)
    return texts, date_time

# Return None if no number plate is detected
    return None, None
