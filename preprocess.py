import cv2
import numpy as np
import os

# load the image we want to de-skew
from skewcurrection.skew_stimation import estimate_skewness, de_skew


def load_image(path):
    try:
        img = cv2.imread(path)
        # perform BRG to gray scale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # invert the color;
        Inverted_Gray = cv2.bitwise_not(gray)

        # covert the gray scale image to binary image;

        binary = cv2.threshold(Inverted_Gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # binary = dilation
        return img, binary
    except:
        print("Could not load the image; provide correct path")
        return None, None


def preprocess_image(image_path):
    img, binary = load_image(image_path)
    if img is None or binary is None:
        return None

    angle = estimate_skewness(binary)
    image = de_skew(img, angle)

    # Check if image is not None
    if image is not None:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (668, 668))  # Resize to your desired size

        image_array = np.array(image)
        image_array = image_array / 255
        return image_array
    else:
        return None


train_path = "/content/train"
val_path = "/content/val"


def prepare_dataset(folder_path):
    X, y = list(), list()
    label_to_int = {"0": 0, "90": 1, "180": 2, "270": 3}
    for folder in os.listdir(folder_path):

        sub_path = folder_path + "/" + folder

        for img in os.listdir(sub_path):
            image_path = os.path.join(sub_path, img)
            processed_image = preprocess_image(image_path)

            X.append(processed_image)
            y.append(label_to_int[folder])

    return np.asarray(X), np.asarray(y)


