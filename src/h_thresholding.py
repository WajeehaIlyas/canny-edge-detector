import numpy as np

def hysteresis_thresholding(image, Th, Tl):
   
    rows, cols = image.shape
    
    output = np.zeros_like(image, dtype=np.uint8)
    
    strong_edges = (image >= Th)
    weak_edges = (image >= Tl) & (image < Th)

    strong_coords = np.argwhere(strong_edges)
    stack = list(strong_coords)
    
    output[strong_edges] = 255
    
    while stack:
        r, c = stack.pop()
        
        # Check all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Check if the neighbor is a weak edge and hasn't been visited yet
                    if weak_edges[nr, nc] and output[nr, nc] == 0:
                        output[nr, nc] = 255 
                        stack.append((nr, nc))

    return output