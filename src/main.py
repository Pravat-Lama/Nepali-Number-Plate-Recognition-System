import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from recognize import recognize_plate

def select_image():
    # Open a file dialog for image selection
    global image_path  # Declare the image_path as a global variable
    image_path = filedialog.askopenfilename()

    # Check if a file was selected
    if image_path:
        # Display the selected image
        display_image(image_path)


def detect_plate():
    # Check if an image is already loaded
    if 'image_path' in globals():
        # Recognize characters on the selected image
        recognized_text, timestamp = recognize_plate(image_path)

        # Print the recognized text and timestamp
        print(f'{recognized_text} detected at {timestamp}')
    else:
        print("Please select an image first.")


def display_image(image_path):
    # Clear any existing image
    if 'image_label' in globals():
        image_label.destroy()

    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image to fit within the window
    image.thumbnail((800, 600))

    # Convert the image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(image)

    # Create a label widget to display the image
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  # Store a reference to the image to prevent garbage collection
    image_label.pack(pady=20)

# Create the main Tkinter window
root = tk.Tk()
root.title("Number Plate Detection")
root.geometry("800x600")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image, width=15, height=2)
select_button.pack(pady=20)

# Create a button to detect the number plate
detect_button = tk.Button(root, text="Detect Plate", command=detect_plate, width=15, height=2)
detect_button.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
