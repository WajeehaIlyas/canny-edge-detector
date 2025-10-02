#Wajeeha Ilyas
#BSCS22022
#Assignment 2

import argparse
import os
import numpy as np
from io_file import image_reading, image_writing
from mask_generation import mask_size, gd_masks, convolution
from gradient import compute_gradient_magnitude, compute_gradient_direction
from nm_suppression import nm_suppression
from h_thresholding import hysteresis_thresholding

def normalize_and_save(image_data, base_name, suffix, output_dir):
    """Normalizes an image and saves it with a descriptive filename."""
    from io_file import image_writing # Assuming this file exists
    
    # Manually normalize the data to 0-255 range
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    if max_val == min_val:
        normalized_image = np.zeros_like(image_data, dtype=np.uint8)
    else:
        normalized_image = 255.0 * (image_data - min_val) / (max_val - min_val)
        normalized_image = normalized_image.astype(np.uint8)
        
    output_path = os.path.join(output_dir, f"{base_name}_{suffix}.jpg")
    image_writing(normalized_image, output_path)

def main(input_file, output_file, sigma, T, Th, Tl):
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading image: {input_file}")
    image = image_reading(input_file)
    if image is None:
        print(f"Error: Unable to read image from '{input_file}'.")
        return
    
    print(f"\nProcessing with sigma={sigma}, T={T}, Th={Th}, Tl={Tl}.")

    # Step 1: Calculate mask size and generate derivative masks
    filter_size = mask_size(sigma, T)
    print(f"Generated a filter of size {filter_size}x{filter_size}.")
    Gx, Gy, scale_factor = gd_masks(sigma, T)
    print("Generated Gaussian derivative masks.")

    # Step 2: Convolve the image with the masks
    fx, fy = convolution(image, Gx, Gy)
    print("Convolved image with Gx and Gy masks.")

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(output_file)

    print("Saving gradients in X and Y directions.")
    normalize_and_save(fx, base_name, "grad_x", output_dir)
    normalize_and_save(fy, base_name, "grad_y", output_dir)

    # Step 3: Compute gradients
    gradient_magnitude = compute_gradient_magnitude(fx, fy, scale_factor)
    gradient_direction = compute_gradient_direction(fx, fy)
    print("Computed gradient magnitude and direction.")

    print("Saving gradient magnitude.")
    normalize_and_save(gradient_magnitude, base_name, "magnitude", output_dir)

    max_magnitude = np.max(gradient_magnitude)
    print(f"Max magnitude: {max_magnitude}")
    print(f"Suggested Th: {0.3 * max_magnitude}, Suggested Tl: {0.15 * max_magnitude}")
    
    # Step 4: Non-Maxima Suppression
    suppressed_image = nm_suppression(gradient_magnitude, gradient_direction)
    print("Applied non-maxima suppression.")

    suppressed_image, quantized_direction = nm_suppression(gradient_magnitude, gradient_direction)
    print("Applied non-maxima suppression and saving quantized orientations.")
    normalize_and_save(quantized_direction, base_name, "orientations", output_dir)

      # Step 5: Hysteresis Thresholding
    final_edges = hysteresis_thresholding(suppressed_image, Th, Tl)
    print("Applied hysteresis thresholding.")
    
    # Step 6: Save the final result
    if final_edges is not None:
        print(f"Saving final edge-detected image to: {output_file}")
        image_writing(final_edges, output_file)
    else:
        print("Error: Hysteresis thresholding failed, no output image to save.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output image file.")
    parser.add_argument("--sigma", type=float, required=True, help="Standard deviation for Gaussian smoothing.")
    parser.add_argument("--T", type=float, required=True, help="Threshold value for mask size.")
    parser.add_argument("--Th", type=float, required=True, help="High threshold for hysteresis.")
    parser.add_argument("--Tl", type=float, required=True, help="Low threshold for hysteresis.")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.sigma, args.T, args.Th, args.Tl)