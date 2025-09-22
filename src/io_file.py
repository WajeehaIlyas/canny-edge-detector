import numpy as np
import cv2
import os

def image_reading(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The file {image_path} does not exist.")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def image_writing(image_data, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.imwrite(output_path, normalized_image)
    print(f"Image saved to {output_path}")
