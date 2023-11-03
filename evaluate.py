import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from preprocess import preprocess_image


def evaluate_model(model_path, image):
    classifier = load_model(model_path)


    image = cv2.resize(image, (668, 668))

    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    predictions = classifier.predict(np.expand_dims(image, axis=0))

    # Find the index of the highest probability in the predicted_label array
    predicted_index = np.argmax(predictions)

    # Define a dictionary that maps indices to labels
    label_to_int = {0: "0", 1: "90", 2: "180", 3: "270"}

    # Convert the index back to the actual label
    actual_label = label_to_int[predicted_index]

    img = Image.fromarray(image)
    img = img.rotate(-int(actual_label), resample=Image.NEAREST, expand=1)

    return img

# if __name__ == "__main__":
#     model_path = "/content/drive/MyDrive/SystemGroupPrj-20231028T181858Z-001/SystemGroupPrj/rotate_model.h5"
#     image_path = "/content/data_34.jpg"

#     predicted_label = evaluate_model(model_path, image_path)
#     print("Predicted label:", predicted_label)
