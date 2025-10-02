import numpy as np

def nm_suppression(magnitude, direction):
    M, N = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    quantized_direction = np.zeros_like(direction)
    
    # 0 degrees: Horizontal (East-West)
    quantized_direction[(direction >= 337.5) | (direction < 22.5) | (direction >= 157.5) & (direction < 202.5)] = 0
    
    # 45 degrees: South-West to North-East
    quantized_direction[(direction >= 22.5) & (direction < 67.5) | (direction >= 202.5) & (direction < 247.5)] = 1
    
    # 90 degrees: Vertical (North-South)
    quantized_direction[(direction >= 67.5) & (direction < 112.5) | (direction >= 247.5) & (direction < 292.5)] = 2
    
    # 135 degrees: North-West to South-East
    quantized_direction[(direction >= 112.5) & (direction < 157.5) | (direction >= 292.5) & (direction < 337.5)] = 3

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q_dir = quantized_direction[i, j]
            
            p1 = 255  # Neighbor 1
            p2 = 255  # Neighbor 2
            
            if q_dir == 0:  # Horizontal (East and West)
                p1 = magnitude[i, j + 1]
                p2 = magnitude[i, j - 1]
            elif q_dir == 1:  # 45 degrees (North-East and South-West)
                p1 = magnitude[i - 1, j + 1]
                p2 = magnitude[i + 1, j - 1]
            elif q_dir == 2:  # Vertical (North and South)
                p1 = magnitude[i - 1, j]
                p2 = magnitude[i + 1, j]
            elif q_dir == 3:  # 135 degrees (North-West and South-East)
                p1 = magnitude[i - 1, j - 1]
                p2 = magnitude[i + 1, j + 1]
            
            if magnitude[i, j] >= p1 and magnitude[i, j] >= p2:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed, quantized_direction