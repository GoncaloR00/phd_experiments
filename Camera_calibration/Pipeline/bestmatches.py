import cv2
from tqdm import tqdm


def find_matching_points(img1, img2, coord_array, epiline_array, window_size=5, l_ratio = 0.8, similarity_func=cv2.norm):
    points_1 = []
    points_2 = []
    for idx, coord in enumerate(tqdm(coord_array)):
        # Initialize best match (lowest distance)
        best_match = None
        second_dist = None
        best_distance = float('inf')
    
        a, b, c = epiline_array[idx]
    
        # Iterate over each point in the epipolar line
        for x in range(0, img2.shape[1]):
            if b != 0:
                y = -1*(a*x + c) / b
                y = int(round(y))  # Ensure y is an integer for indexing
            else:
                continue  # If line is vertical, skip this iteration
    
            # Check if y is within image bounds
            if y < 0 or y >= img2.shape[0]:
                continue
            
            a1 = coord[1]-window_size
            b1 = coord[1]+window_size
            c1 = coord[0]-window_size
            d1 = coord[0]+window_size
            a2 = y-window_size
            b2 = y+window_size
            c2 = x-window_size
            d2 = x+window_size
    
            if a1 < 0 or c1 <0 or a2 <0 or c2 <0 or b1 > img1.shape[0] or d1 > img1.shape[1] or b2 > img2.shape[0] or d2 > img2.shape[1]:
                continue

            # Compute similarity measure
            distance = similarity_func(img1[a1:b1, c1:d1],
                                    img2[a2:b2, c2:d2], cv2.NORM_L1)
            # Update best match if better
            if distance < best_distance:
                second_dist = best_distance
                best_distance = distance
                best_match = (x, y)
        # Remove invalids
        if second_dist == float('inf') or second_dist is None:
            best_match = None
        # Filter results
        elif not(best_distance < l_ratio*second_dist):
            best_match = None
        # Append valid results
        if not(best_match is None):
            points_1.append((coord[0], coord[1]))
            points_2.append(best_match)
    return points_1, points_2