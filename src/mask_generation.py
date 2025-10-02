import numpy as np

def mask_size(sigma, T):
    if T <= 0 or T >= 1:
        raise ValueError("The threshold T must be between 0 and 1.")
    
    sHalf = np.round(np.sqrt(-np.log(T) * 2 * (sigma ** 2)))
    N = 2 * sHalf + 1
    return int(N)

def coordinate_grid(N):
    if N % 2 == 0 or N <= 0:
        raise ValueError("N must be a positive odd integer.")
    
    half_size = N // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    X, Y = np.meshgrid(x, y)
    
    return X, Y

def gd_masks(sigma, T):
    N = mask_size(sigma, T)
    X, Y = coordinate_grid(N)
    
    exponent = -(X**2 + Y**2) / (2 * sigma ** 2)
    Gx = -X * np.exp(exponent) / (2 * np.pi * sigma ** 4)
    Gy = -Y * np.exp(exponent) / (2 * np.pi * sigma ** 4)
    
    scale_factor = 255.0 / np.max(np.abs(np.concatenate([Gx.flatten(), Gy.flatten()])))
    Gx_scaled = np.round(Gx * scale_factor)
    Gy_scaled = np.round(Gy * scale_factor)
    
    return Gx_scaled.astype(np.int32), Gy_scaled.astype(np.int32), scale_factor

def convolution(image, Gx, Gy):
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    image_height, image_width = image.shape
    mask_height, mask_width = Gx.shape
    
    pad_height = mask_height // 2
    pad_width = mask_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    
    fx = np.zeros_like(image, dtype=np.float32)
    fy = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + mask_height, j:j + mask_width]
            fx[i, j] = np.sum(region * Gx)
            fy[i, j] = np.sum(region * Gy)
    
    return fx, fy