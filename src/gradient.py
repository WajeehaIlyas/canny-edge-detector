import numpy as np

def compute_gradient_magnitude(fx, fy, scale_factor):
    magnitude = np.sqrt(fx**2 + fy**2)
    magnitude /= scale_factor
    
    return magnitude

def compute_gradient_direction(fx, fy):
    direction_rad = np.arctan2(fy, fx)
    
    direction_deg = np.degrees(direction_rad)
    
    direction_deg += 180
    
    return direction_deg