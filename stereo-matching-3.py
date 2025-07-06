# Author: Wiktoria Marczyk
# Lab 3 - Stereo Matching

import math
import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm

# ---------- PARAMETERS ----------

image_cones_left = "Data/cones/cones-left/im2.png"
image_cone_right = "Data/cones/cones-right/im6.png"
ref_disp_cones = "Data/cones/disp2.png"
pixel_cut = 65

stereo_calib_params_path = "stereo_calib_params_leftright.json"
image_stereo_left = "left-55.png"
image_stereo_right = "right-55.png"

# ---------------------------------

# ---------- FUNCTIONS ----------

def stereoBM(img_left_path, img_right_path, block_size=11, num_disparities=64):
    """
    Calculates the disparity map using the StereoBM algorithm.
    :param img_left_path: Path to the left image.
    :param img_right_path: Path to the right image.
    :param ref_img: Reference disparity map.
    :param block_size: Size of the window used to match pixels. It must be odd and between 5 and 255.
    :param num_disparities: The number of disparities. It is the number of pixels that will be compared in the left and right images. It must be divisible by 16.
    """
    print("\n===== STEREO BM =====")

    # Load rectified pair of images to be used for disparity map calculation
    left_img = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
    right_img = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise ValueError("Couldn't load images")
    if left_img.shape != right_img.shape:
        raise ValueError("Images must have the same shape")


    # Create a stereo block matching object
    stereo = cv.StereoBM.create(numDisparities=num_disparities, blockSize=block_size)
    # NOTE: Code returns a 16-bit fixed-point disparity map where each disparity value has 4 fractional bits
    # To get the floating-point disparity map, divide the disparity values by 16

    # Compute disparity map 
    disparity_map = stereo.compute(left_img, right_img)
    # Divide disparity values by 16
    disparity_map = disparity_map.astype(np.float32) / 16.0

    # Cut 65 pixels from the left side of the disparity map cause it's all zeros
    disparity_map = disparity_map[:, pixel_cut:]

    return disparity_map


def stereoSGBM(img_left_path, img_right_path, min_disparity=0, block_size=5, num_disparities=64):
    """
    Calculates the disparity map using the StereoSGBM (Semi-Global Block Matching) algorithm.
    :param img_left_path: Path to the left image.
    :param img_right_path: Path to the right image.
    :param ref_img: Reference disparity map.
    :param min_disparity: Minimum possible disparity value. Normally, it is 0.
    :param block_size: Size of the block used for matching. It must be an odd number >=1. Normally, it should be somewhere in the 3...11 range.
    :param num_disparities: Maximum disparity minus minimum disparity. The value is always greater than zero. Must be divisible by 16.
    """
    print("\n===== STEREO SGBM =====")

    # Load rectified pair of images to be used for disparity map calculation
    left_img = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
    right_img = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise ValueError("Couldn't load images")
    if left_img.shape != right_img.shape:
        raise ValueError("Images must have the same shape")


    # according to the OpenCV documentation:
    # for p1: 8*number_of_image_channels*blockSize*blockSize 
    # for p2: 32*number_of_image_channels*blockSize*blockSize
    # The algorithm requires P2 > P1
    p1 = 8*3*block_size*block_size  # Penalty on the disparity change by plus or minus 1 between neighbor pixels
    p2 = 16*3*block_size*block_size # Penalty on the disparity change by more than 1 between neighbor pixels 
    disp12_max_diff = 1             # Max disparity difference between left-right checks. Set it to a non-positive value to disable the check.
    uniqueness_ratio = 5            # Percentage of uniqueness for the disparity match. Normally, a value within the 5-15 range is good enough.
    speckle_window_size = 100       # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it in the 50-200 range.
    speckle_range = 1               # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough

    # Create a stereo semi-global block matching object
    stereo = cv.StereoSGBM.create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range
    )
    
    # Compute disparity map 
    disparity_map = stereo.compute(left_img, right_img)
    disparity_map = disparity_map.astype(np.float32) / 16.0

    # Cut 65 pixels from the left side of the disparity map cause it's all zeros
    disparity_map = disparity_map[:, pixel_cut:]

    return disparity_map


def compare_disparity_maps(disp_map, ref_map):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Computed Disparity")
    plt.imshow(disp_map, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Reference Disparity")
    plt.imshow(ref_map, cmap='gray')
    plt.colorbar()

    plt.show()


# Decode from colors to disparity values (for 'Cones' and 'Teddy': 1-255 to 0.25-63.75)
def decode_ref_disparity_map(ref_disp):

    ref_disp = ref_disp.astype(np.float32)
    ref_disp[ref_disp > 0] = (ref_disp[ref_disp > 0] / 255.0) * 63.5 + 0.25
    
    return ref_disp


# Visualize disparity maps errors using the reference disparity map
def visualize_disparity_errors(disp_map, ref_disp):
    
    # calculate the error map
    error_map = np.abs(disp_map - ref_disp)

    mse =  np.mean((disp_map - ref_disp) ** 2)
    print("Mean Squared Error: ", mse)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Disparity Map")
    plt.imshow(disp_map, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Reference Disparity")
    plt.imshow(ref_disp, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Disparity Error Map")
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()

    plt.show()


# ----- Template matching functions -----

def tmf_sqdiff(template, roi):
    return np.sum(pow((template - roi), 2))

def tmf_sqdiff_norm(template, roi):
    return np.sum(pow((template - roi), 2)) / math.sqrt(np.sum(pow(template, 2)) * np.sum(pow(roi, 2)))

# ---------------------------------


def calculate_disparity_map(img_left_path, img_right_path, block_size=5, num_disparities=64):

    # Load rectified pair of images to be used for disparity map calculation
    img_left = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
    img_right = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        raise ValueError("Couldn't load images)")
    if img_left.shape != img_right.shape:
        raise ValueError("Images must have the same shape")

    height, width = img_left.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block = block_size // 2

    # Pad images to handle pixels on the borders
    left_padded = cv.copyMakeBorder(img_left, half_block, half_block, half_block, half_block, cv.BORDER_REPLICATE, value=0)
    right_padded = cv.copyMakeBorder(img_right, half_block, half_block, half_block, half_block, cv.BORDER_REPLICATE, value=0)

    for y in tqdm(range(half_block, height + half_block)):
        for x in range(half_block, width + half_block):
            # Get block from left image
            left_block = left_padded[y - half_block:y + half_block + 1, x - half_block:x + half_block + 1]

            min_tmf_cost = float('inf')
            best_disparity = 0

            # For each disparity level compute template matching function cost
            for d in range(num_disparities):
                x_block_start = x - d - half_block
                
                if x_block_start < 0:
                    continue

                # Get block from right image
                right_block = right_padded[y - half_block:y + half_block + 1, x_block_start:x_block_start + block_size]

                # Compute template matching function cost
                tmf_value = tmf_sqdiff(left_block, right_block)

                if tmf_value >= min_tmf_cost:
                    continue

                # Update best disparity
                best_disparity = d
                min_tmf_cost = tmf_value

            disparity_map[y - half_block, x - half_block] = best_disparity

    # Normalize disparity map for visualization
    disparity_map = cv.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)

    return disparity_map


# ----- Rectify and calculate disparity map -----

# Function that reads the stereo calibration parameters from a file
def read_stereo_params_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        mtx1 = np.array(data["left_camera_matrix"])
        dist1 = np.array(data["left_distortion_coefficients"])
        mtx2 = np.array(data["right_camera_matrix"])
        dist2 = np.array(data["right_distortion_coefficients"])
        R = np.array(data["rotation_matrix"])
        T = np.array(data["translation_vector"])
        E = np.array(data["essential_matrix"])
        F = np.array(data["fundamental_matrix"])
        baseline = data["baseline"]
        fov1 = data["left_camera_fov"]
        fov2 = data["right_camera_fov"]

        print("Stereo calibration parameters loaded from: " + path)
    return mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov1, fov2


# Function responsible for drawing the region of interest and epipolar lines on the rectified images
def draw_roi_and_epilines(img_left, img_right, roi1, roi2):
    img_left_copy = img_left.copy()
    img_right_copy = img_right.copy()

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    line_thickness_roi = 3
    line_thickness_lines = 2
    lines_count = 20

    # draw the ROIs
    color = (0, 255, 0)
    cv.rectangle(img_left_copy, (x1, y1), (x1 + w1, y1 + h1), color, line_thickness_roi)
    cv.rectangle(img_right_copy, (x2, y2), (x2 + w2, y2 + h2), color, line_thickness_roi)

    # concatenate the images
    rectified_pair = np.hstack((img_left_copy, img_right_copy))

    # draw epipolar lines
    new_height, new_width = rectified_pair.shape[:2]
    color = (0, 0, 255)
    y_coords = np.linspace(0, new_height - 1, lines_count).astype(int)
    for y in y_coords:
        cv.line(rectified_pair, (0, y), (new_width - 1, y), color, line_thickness_lines)

    # save the image with ROIs and epipolar lines drawn
    cv.imwrite("rectified_with_ROI.png", rectified_pair)


def rectify_stereo_images(img_left_path, img_right_path, mtx1, dist1, mtx2, dist2, R, T):
    print("\n===== Rectifying stereo images =====")
    
    img_left = cv.imread(img_left_path)
    img_right = cv.imread(img_right_path)

    crop_img = False
    flags = cv.CALIB_ZERO_DISPARITY
    # alpha -  the rectified images are zoomed and shifted so that mainly valid pixels are visible
    # alpha = 0.6 - cam1 cam4 Mono & Stereo
    # alpha = 0.5 - cam2 cam3 Mono
    # alpha = 0.95 - cam2 cam3 Stereo
    alpha = 0.95
    if alpha > 0.5:
        crop_img = True
        print("Alpha > 0.5 - need to crop the images")

    gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    print("Original image size", image_size)

    # get rectification matrices, projection matrices and disparity-to-depth mapping matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, image_size, R, T, flags=flags, alpha=alpha)

    # get the rectification maps
    map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, image_size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, image_size, cv.CV_32FC1)

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    # remap the images
    img_left_rectified = cv.remap(img_left, map1x, map1y, cv.INTER_LINEAR)
    img_right_rectified = cv.remap(img_right, map2x, map2y, cv.INTER_LINEAR)


    # crop the images
    if crop_img == True:

        x_left_up = min(x1, x2)
        y_left_up = min(y1, y2)

        x_right_down = max(x1 + w1, x2 + w2)
        y_right_down = max(y1 + h1, y2 + h2)

        img_left_rectified = img_left_rectified[y_left_up : y_right_down, x_left_up : x_right_down]
        img_right_rectified = img_right_rectified[y_left_up : y_right_down, x_left_up : x_right_down]

        roi1 = (x1 - x_left_up, y1 - y_left_up, w1, h1)
        roi2 = (x2 - x_left_up, y2 - y_left_up, w2, h2)

        print ("ROI1: ", roi1, "ROI2: ", roi2) 

        # check if the rectification was successful
        draw_roi_and_epilines(img_left_rectified, img_right_rectified, roi1, roi2)

    return img_left_rectified, img_right_rectified


def rectify_and_calc_disparity_map(img_left_path, img_right_path, stereo_params_path, min_disparity=0, block_size=5, num_disparities=64):
    """
    Calculates the disparity map using the StereoSGBM algorithm.
    :param img_left_path: Path to the left image.
    :param img_right_path: Path to the right image.
    :param stereo_params_path: Path to the stereo calibration parameters file.
    :param min_disparity: Minimum possible disparity value. Normally, it is 0.
    :param block_size: Size of the window used to match pixels. It must be odd and between 5 and 255.
    :param num_disparities: The number of disparities. It is the number of pixels that will be compared in the left and right images. It must be divisible by 16.
    """
    print("\n===== RECTIFY IMAGES AND CALCULATE DISPARITY MAP =====")

    # Load stereo calibration parameters
    mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov1, fov2 = read_stereo_params_from_json(stereo_params_path)

    # Load pair of images to be used for disparity map calculation
    left_img = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
    right_img = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise ValueError("Couldn't load images")
    if left_img.shape != right_img.shape:
        raise ValueError("Images must have the same shape")

    # Rectify images
    img_left_rectified, img_right_rectified = rectify_stereo_images(img_left_path, img_right_path, mtx1, dist1, mtx2, dist2, R, T)

    img_left_rectified_path = "rectified_" + img_left_path
    img_right_rectified_path = "rectified_" + img_right_path

    # Save the rectified images
    cv.imwrite(img_left_rectified_path, img_left_rectified)
    cv.imwrite(img_right_rectified_path, img_right_rectified)

    # Calculate disparity map using the StereoSGBM algorithm
    disparity_map = stereoSGBM(img_left_rectified_path, img_right_rectified_path, min_disparity, block_size, num_disparities)

    img_left_rectified = cv.imread(img_left_rectified_path, cv.IMREAD_GRAYSCALE)
    img_right_rectified = cv.imread(img_right_rectified_path, cv.IMREAD_GRAYSCALE)

    # Visualize left img, right img and disparity map
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Left Image")
    plt.imshow(img_left_rectified, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Right Image")
    plt.imshow(img_right_rectified, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Disparity Map")
    plt.imshow(disparity_map, cmap='gray')
    plt.colorbar()

    plt.show()

    return disparity_map

# ---------------------------------

# ----- MAIN -----

print("\n===== STARTING PROGRAM - STEREO MATCHING =====")  

# Load reference disparity map
ref_map = cv.imread(ref_disp_cones, cv.IMREAD_GRAYSCALE)
# Decode from colors to disparity values
ref_map = decode_ref_disparity_map(ref_map)
# Cut 65 pixels from the left side of the reference disparity map to match size with computed disparity maps
ref_map_cut = ref_map[:, pixel_cut:]

disp_map_bm = stereoBM(image_cones_left, image_cone_right)
disp_map_sgbm = stereoSGBM(image_cones_left, image_cone_right)

print("\n===== VISUALIZING DISPARITY ERRORS - BM =====")
visualize_disparity_errors(disp_map_bm, ref_map_cut)
print("\n===== VISUALIZING DISPARITY ERRORS - SGBM =====")
visualize_disparity_errors(disp_map_sgbm, ref_map_cut)

print("\n===== CUSTOM STEREO MATCHING FUNCTION =====")
disp_map = calculate_disparity_map(image_cones_left, image_cone_right)
disp_map_cut = disp_map[:, pixel_cut:]
compare_disparity_maps(disp_map_cut, ref_map_cut)
visualize_disparity_errors(disp_map_cut, ref_map_cut)

print("\n===== RECTIFY AND CALCULATE DISPARITY MAP =====")
disparity_map = rectify_and_calc_disparity_map(image_stereo_left, image_stereo_right, stereo_calib_params_path)

# ---------------------------------