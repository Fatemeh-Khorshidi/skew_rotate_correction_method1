import argparse
import os
import time
from preprocess import load_image
from skewcurrection.skew_stimation import estimate_skewness, de_skew
from evaluate import evaluate_model


def main(img_dir, new_dir, model_path):
    os.makedirs(new_dir, exist_ok=True)
    count = 0
    start = time.time()
    for image in os.listdir(img_dir):
        if image.endswith(("jpg", "png")):
            image_dir = os.path.join(img_dir, image)
            print(image_dir)
            img, binary = load_image(image_dir)
            if img is None or binary is None:
                return None

            angle = estimate_skewness(binary)
            rotated_image = de_skew(img, angle)
            rotated = evaluate_model(model_path, rotated_image)
            count += 1
            new_image_path = os.path.join(new_dir, f'{image}.jpg')

            rotated.save(new_image_path)

    end = time.time()
    total_time = end - start
    print("total time", total_time)


image_dir = './samples/test'
model_path = './rotate_model.h5'
save_dir = 'app/final_result'

if __name__ == "__main__":

    main(image_dir, save_dir ,model_path)
