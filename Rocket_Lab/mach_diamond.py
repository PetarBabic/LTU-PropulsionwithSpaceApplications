import numpy as np
import cv2 as cv
from skimage import measure
import csv
import pandas as pd

cap = cv.VideoCapture('Test.cam1.avi')

Duration = 2 * 60 + 28.73
fps = 30
burn_start = 28.6666667 * 30
burn_end = 33 * 30

current_frame = burn_start

cap.set(cv.CAP_PROP_POS_FRAMES, burn_start)

x,y,h,w = 245,235,35,35

frame_number = int(burn_start)

with open("mach_diamonds.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["frame", "distance"])
    
    while cap.isOpened() and current_frame <= burn_end:
        ret, frame = cap.read()
        current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # The region of interest
        crop_frame = frame[y:y+h, x:x+w]
        # Getting the monochrome image
        gray = cv.cvtColor(crop_frame, cv.COLOR_BGR2GRAY)
        # Blurring the image to reduce the noise
        blur = cv.GaussianBlur(gray,(13,13),0)
        # Binarizing the image to find the mach diamond
        th = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,3)
        # "Expanding the regions" to additionally supress the noise
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
        th = cv.morphologyEx(th, 
                                cv.MORPH_OPEN,
                                kernel,
                                iterations=2)
        # Inverting the binarized image so that measure can find the regions (it searches for only white)
        th = cv.bitwise_not(th)
        
        # Finding the regions
        labels = measure.label(th, connectivity=1, background=0)
        regions = measure.regionprops(labels)
        # Validating the found regions
        valid_regions = [r for r in regions if 5 < r.area < 2000]
        if valid_regions:
            largest_region = max(valid_regions, key=lambda r: r.area)

            # Getting the left most pixel of the region
            y_ = largest_region.bbox[0] 
            x_ = largest_region.bbox[1] 

            # Data visualization
            # gray[y_, x_] = 255   # mark centroid

            # Verti = np.concatenate((blur, th, gray), axis=0)
            
            # cv.imshow('frame', Verti)
            # key = cv.waitKey(500)

            # if key == 32:
            #     cv.waitKey()
            # elif key == ord('q'):
            #     break

            red_cap_length_px = 43.01
            red_cap_length_m = 0.8e-2
            pixel_scale = red_cap_length_m / red_cap_length_px  # m/px

            global_xy = np.array([x_ + x, y_ + y])

            # Top-right, bottom-left of nozzle
            # Format: [x1, y1, x2, y2]
            nozzle_line = np.array([237, 230, 225, 264])

            # Extract points
            p1 = nozzle_line[0:2]  # [x1, y1]
            p2 = nozzle_line[2:4]  # [x2, y2]

            # Distance from point to line segment formula
            line_vec = p2 - p1
            point_vec = global_xy - p1
            distance = np.linalg.norm(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)

            # Convert to meters
            writer.writerow([frame_number, distance * pixel_scale])

        frame_number += 1
 
cap.release()
cv.destroyAllWindows()