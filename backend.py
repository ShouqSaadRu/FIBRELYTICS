import cv2
import numpy as np
from skimage import feature, measure, segmentation
from scipy import ndimage
import pandas as pd



def apply_mean_shift(image, params):
    sp = params["sp"]
    sr = params["sr"]
    return cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)



def apply_gaussian_blur(image, params):
    blur_size = params["blur_size"]
    return cv2.GaussianBlur(image, (blur_size, blur_size), 0)



def apply_clahe(gray, params):
    clip_limit = params["clip_limit"]
    tile_grid_size = params["tile_grid_size"]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)



def threshold_image(gray_clahe, params):
    blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary



def Euclidean_Distance_Map(binary):
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5)



def Local_Maxima_Search(distance_map, binary, params):
    coordinates = feature.peak_local_max(
    distance_map, min_distance=params['min_distance'], labels=binary
                                                )
    # Create a binary mask from the coordinates
    local_max = np.zeros_like(distance_map, dtype=bool)
    local_max[tuple(coordinates.T)] = True
    return local_max



def Connected_Components(local_max):
    markers, _ = ndimage.label(local_max)
    return markers


def Watershed(binary, markers):
    labels = segmentation.watershed(binary, markers, mask=binary)
    return labels


def Contour_Center_Location_extraction(labels):
    contours = []
    # Loop over each label (ignoring background label 0)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        # Create a binary mask for the current label
        mask_label = np.uint8(labels == label_id)

        # Find contours for the current label
        cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours.append(cnts[0])  # Assuming one contour per object

    props = measure.regionprops(labels)
    fibre_centers = [prop.centroid for prop in props]
    return fibre_centers, contours
    

def remove_close_points(points, params):
    filtered = []
    for p in points:
        if all(np.linalg.norm(np.array(p) - np.array(f)) > params['threshold'] for f in filtered):
            filtered.append(p)
    return filtered



def average_radius_from_contour_calculations(image, contour, center, num_directions=8, tolerance=1.0):
    
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False) #compute 8 directions around a circle
    yc, xc = center
    distances = [] #stores each ray length.
    ray_endpoints = []  #saves hit points on the contour.

    contour = contour.reshape(-1, 2)  



    #loop over each direction and dx, dy are components of unit vector in this direction
    for theta in angles:
        dx = np.cos(theta)
        dy = np.sin(theta)
        r = 0 #distance from center that is incremented to step out along ray




        #step pixel by pixel along the ray
        #calculate the (x, y) coordinates of the current step on the ray.
        while True:
            x = int(round(xc + dx * r))
            y = int(round(yc + dy * r))




            #If the ray goes outside the image, stop
            if (x < 0 or y < 0 or y >= image.shape[0] or x >= image.shape[1]):
                break
            
            
            
            
            #check for hit on contour
            #If the current pixel matches any point on the contour, it's a hit!
            #Compute the distance from center to this contour point.
            #Save the distance and the (x, y) hit point.
            #Then stop this ray and go to the next one.
            #pointPolygonTest checks how close a point is to a contour and returns 0 if the point is on the contour. But we can relax the condition and accept points near the contour using a tolerance.
            dist_to_contour = cv2.pointPolygonTest(contour, (x, y), True)
            if abs(dist_to_contour) <= tolerance: 
                distances.append(np.sqrt((x - xc)**2 + (y - yc)**2)) #yc and xc are centers while x and y are the contour
                ray_endpoints.append((x, y))  # store hit point
                break

            #Increase r to keep moving out along the ray until a hit is found.
            r += 1


    #only return the average radius if at least 4 hits were found (for reliability)
    if len(distances) >= 1:
        return np.mean(distances), ray_endpoints
    else:
        return None, []


def average_radius_from_contour(image ,filtered_centers, contours):
    accepted_centers = []
    average_radius = []
    all_rays = []  


    for center, contour in zip(filtered_centers, contours):
        avg_r, rays = average_radius_from_contour_calculations(image ,contour, center)
        if avg_r is not None:
            accepted_centers.append(center)
            average_radius.append(avg_r)
            all_rays.append((center, rays))  # store rays for visualization

    return accepted_centers, average_radius, all_rays




def calculate_fiber_volume_fraction(image, contours, mask=None):
    contour_area = [cv2.contourArea(np.array(cnt, dtype=np.int32)) for cnt in contours]
    total_fiber_area = sum(contour_area) # or cv2.countNonZero(binary) both work
    if mask is None: 
    #get total tape area
        image_area = image.shape[0] * image.shape[1] 
    else:
        image_area = cv2.countNonZero(mask)
    return (total_fiber_area / image_area) * 100, contour_area






def process_image_pipeline(image, params, mask=None):
    result = {}

    result["Original"] = image.copy()
    
    # Step 1: Mean Shift
    mean_shifted = apply_mean_shift(image, params)
    result["Mean Shift"] = mean_shifted
    print(mean_shifted)
    # Step 2: Gaussian Blur
    blurred = apply_gaussian_blur(mean_shifted, params)
    result["Blurred"] = blurred
    
    # Step 3: Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    result["Gray scale"] = gray
    print("This is gray scaling")
    print(gray)
    # Step 4: CLAHE
    gray_clahe = apply_clahe(gray, params)

    # Step 5: Threshold
    binary = threshold_image(gray_clahe, params)
    result["Threshold"] = binary
    print("This is threshold")
    print(binary)
    # Step 6: Euclidean Distance Map
    distance_map = Euclidean_Distance_Map(binary)
    
    # Step 7: Local Maxima Search
    local_max = Local_Maxima_Search(distance_map, binary, params)
    
    # Step 8: Connected Components
    markers = Connected_Components(local_max)
    
    # Step 9: Watershed
    labels = Watershed(binary, markers)
    result["Watershed"] = labels
    
    # Step 10: Contour and Center Location extraction
    fibre_centers, contours = Contour_Center_Location_extraction(labels)
    result["Contours"] = contours
    # Step 11: Proximity - neighbour check
    filtered_centers = remove_close_points(fibre_centers, params)
    result["Fibre Centers"] = filtered_centers
    
    # Step 12: Average Radius From Contour
    accepted_centers, average_radius, all_rays = average_radius_from_contour(image, filtered_centers, contours)
    result["accepted_centers"] = accepted_centers
    result["average_radius"] = average_radius
    result["all_rays"] = all_rays

    print("Number of detected fibres:", len(filtered_centers))
    for i, center in enumerate(filtered_centers):
        print(f"Fibre {i+1}: (y={center[0]:.2f}, x={center[1]:.2f})")
        
    # Step 13: Fiber Volume Fraction
    v_fv , contour_area= calculate_fiber_volume_fraction(image ,contours, mask)
    result["V_fv"] = v_fv
    result["contour_area"]= contour_area
    return result


def save_fiber_data(result, save_path):
    accepted_centers = result["accepted_centers"]
    average_radius = result["average_radius"]
    contour_area = result["contour_area"]
    
    fiber_ids = list(range(1, len(accepted_centers) + 1))
    xs = [x for y, x in accepted_centers]
    ys = [y for y, x in accepted_centers]
    circular_areas = [np.pi * r**2 for r in average_radius]

    df = pd.DataFrame({
        'Fiber_ID': fiber_ids,
        'X': [round(x, 6) for x in xs],
        'Y': [round(y, 6) for y in ys],
        'Average_Radius': [round(r, 6) for r in average_radius],
        'Circular_Area': [round(a, 1) for a in circular_areas],
        'Contour_Area': contour_area
    })
    
    df.to_csv(save_path, index=False)
