import math
from typing import Tuple, Union
from deskew import determine_skew
from pytesseract import Output
import pytesseract
import argparse
import numpy as np
from PIL import Image
import cv2
import imutils
import os

import time

dir = '/content/'
new_dir = '/content/'
os.makedirs(new_dir, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Rotate an image by a specified angle.

    Args:
        image (np.ndarray): The input image to be rotated.
        angle (float): The rotation angle in degrees.
        background (Union[int, Tuple[int, int, int]]): The background color for areas outside the original image.

    Returns:
        np.ndarray: The rotated image.

    This function rotates the input image by the specified angle while maintaining the original image's dimensions. It uses OpenCV to perform the rotation and allows specifying a background color for areas that become visible due to the rotation.

    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def rotate_correction(image_path, new_path):
    """
    Rotate and correct images in a given directory.

    Args:
        image_path (str): The path to the directory containing input images.
        new_path (str): The path to the directory where rotated and corrected images will be saved.

    Returns:
        list of np.ndarray: List of rotated and corrected images.

    This function iterates through image files in the specified directory, detects the rotation angle, rotates the images, and saves them to a new directory. It also corrects the skew of the images before rotation and saves them as JPEG files.

    """
    for img in os.listdir(image_path):
        if img.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            start_time = time.time()
            image_dir = os.path.join(image_path, img)
            image = cv2.imread(image_dir)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
            angle_ = results["rotate"]
            rotated_ = imutils.rotate_bound(image, angle=angle_)
            grayscale = cv2.cvtColor(rotated_, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(grayscale)
            rotated = rotate(rotated_, angle, (0, 0, 0))
            new_image_path = os.path.join(new_path, f'{img}')
            rotated_image = Image.fromarray(rotated)
            # cv2.imwrite(new_image_path, rotated_image)
            rotated_image.save(new_image_path, format="JPEG")
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f"Iteration for {img} took {elapsed_time:.2f} seconds")
            print("done")

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="rotate and skew currection")
    parser.add_argument("--image_path", required=True, help="Path to the input scaned document image")
    parser.add_argument("--save_path", required=True, help="Path to the output scaned document image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    rotate_correction(args.image_path, args.save_path)
