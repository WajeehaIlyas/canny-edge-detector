# canny-edge-detector

# project components
main.py: The main entry point of the program. It orchestrates the entire Canny pipeline, taking command-line arguments and calling functions from other modules.

gradient.py: Contains functions to compute the gradient magnitude and direction from the convolved images.

nm_suppression.py: Implements the non-maxima suppression step to thin edges.

h_thresholding.py: Implements the hysteresis thresholding step to finalize the edge map.

mask_generation.py: Handles the creation of Gaussian derivative masks and the convolution process.

io.py: Contains functions for reading and writing images in a simple text-based format, adhering to the no-library constraint.

# command line arguments

--input_file: Path to the input image file.

--output_file: Path where the final output image will be saved.

--sigma: Standard deviation of the Gaussian filter. A higher value results in more blurring.

--T: Threshold value for determining the size of the Gaussian mask.

--Th: High threshold for hysteresis. 

--Tl: Low threshold for hysteresis. 

# example command 
python3 src/main.py --input_file data/circle.jpg --output_file results/final_edges.jpg --sigma 1.0 --T 0.3 --Th 0.3 --Tl 0.15