import numpy as np
import cv2
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)


def Skewed_angle(x1, y1, x2, y2):
    """
    Calculate the angle (in degrees) between two points (x1, y1) and (x2, y2) relative to the positive x-axis.
    Returns 0 if a ZeroDivisionError occurs.

    Parameters:
    - x1 (float): x-coordinate of the first point.
    - y1 (float): y-coordinate of the first point.
    - x2 (float): x-coordinate of the second point.
    - y2 (float): y-coordinate of the second point.

    Returns:
    - angle (float): The angle between the two points in degrees.
    """
    try:
        y = (y2 - y1)  # numerator of arctan
        x = (x2 - x1)  # denominator of arctan

        return (np.rad2deg(np.arctan2(y, x)))

    except ZeroDivisionError:
        return 0


def estimate_skewness(binary_image):
    """
    Estimate the skewness angle of a binary image using the Hough line transform.

    Parameters:
    - binary_image (numpy.ndarray): A binary image (e.g., after edge detection).

    Returns:
    - skewness_angle (float): The estimated skewness angle in degrees.
    """
    # Classic straight-line Hough transform
    h, theta, d = hough_line(binary_image)

    lines = hough_line_peaks(h, theta, d)  # gives the highest voted lines returned by hough_line function

    # take horizontal lines only
    indices = np.where(abs(np.rad2deg(lines[1])) > 45)  # array indices of horizontal lines
    # horizontal_lines= (np.take (lines[1], indices),np.take (lines[2], indices))
    horizontal_lines = (lines[1][indices], lines[2][indices])

    line_angle = []
    if indices[0].size > 0:
        for angle, dist in zip(*horizontal_lines):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - binary_image.shape[1] * np.cos(angle)) / np.sin(angle)
            line_angle.append(Skewed_angle(0, y0, binary_image.shape[1], y1))

    if len(line_angle):
        return np.mean(line_angle)
    else:
        return 0


# de-skew the image
def de_skew(img, angle):
    """
    De-skew an image by rotating it by a specified angle.

    Parameters:
    - img (numpy.ndarray): The input image to be de-skewed.
    - angle (float): The angle (in degrees) by which to rotate the image for de-skewing.

    Returns:
    - de_skewed_img (numpy.ndarray): The de-skewed image.
    """
    rows, cols, _ = img.shape
    # print ("Angle : ", angle)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    # fig2= plt.figure (figsize=(15,15))

    # plt. imshow (dst[:,:,::-1])
    # cv2.imwrite('corrected_image.jpg',dst)
    return dst
