import cv2
import os
import numpy as np


def grayscale_CAPTCHA_image(captcha_image):
    "Grayscales CAPTCHA image."""
    return cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)

def threshold_CAPTCHA_image(captcha_image_grayscaled):
    """Thresholds CAPTCHA image."""
#     return cv2.threshold(captcha_image_grayscaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return cv2.threshold(captcha_image_grayscaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def dilate_characters(binary_image):
    """slightly expands the characters."""
    kernel = np.ones((2,2), np.uint8) 
    return cv2.dilate(binary_image, kernel,iterations = 1)

def find_CAPTCHA_contours(captcha_image_thresholded):
    """Compute the contours of characters in the CAPTCHA image."""
    return cv2.findContours(captcha_image_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

def compute_bounding_rectangles(contours):
    """Computes the bounding rectangles of the contours."""
    return list(map(cv2.boundingRect,contours))



def show_bounding_rectangles(rectangles, image):
    """Shows the bounding rectangles of contours on the image."""
    for rect in rectangles:
        x,y,w,h = rect
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

def split_fat_rectangles(rectangles):
    """Splits fat rectangles into two rectangles."""
    letter_bounding_rectangles = []
    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        print
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_bounding_rectangles.append((x, y, half_width, h))
            letter_bounding_rectangles.append((x + half_width, y, half_width, h))
        else:
            letter_bounding_rectangles.append(rectangle)
    return letter_bounding_rectangles

def sort_bounding_rectangles(rects): 
    """Sorts bounding rectangles by x coordinate."""
    return(sorted(rects, key = lambda x: float(x[0]))) 
def get_character_images(rectangles, image):
    """Extracts the characters defined by bounding rectangles."""
    char_images = []
    for rect in rectangles:
            x, y, w, h = rect
            char_image = image[y - 1 : y + h + 1, x - 1 : x + w + 1]
            char_images.append(char_image)
    return char_images
