#Wajeeha Ilyas
#BSCS22022
#Assignment 2

import argparse
import os
from io_file import image_reading, image_writing
from mask_generation import mask_size, gd_masks, convolution
from gradient import compute_gradient_magnitude, compute_gradient_direction

def main(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading image: {input_path}")
    image = image_reading(input_path)
    
    if image is not None:
        print(f"Saving processed image to: {output_path}")
        image_writing(image, output_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output image file.")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file)