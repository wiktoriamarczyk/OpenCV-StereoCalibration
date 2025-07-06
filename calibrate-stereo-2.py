# Author: Wiktoria Marczyk
# Lab 2 - Stereo Calibration

import math
import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json

# ---------- PARAMETERS ----------

# flags for enabling/disabling calibration processes
CAMERAS_CALIBRATE = True
STEREO_CALIBRATE = True
READ_STEREO_CALIB_PARAMS = False

# define the dimensions of the chessboard (number of inner corners per row and column)
chessboard_dim = (7, 10)

# define the size of a single chessboard square in mm
single_square = 28.67

# define method to find corners, True for cornerSubPix, False for findChessboardCorners
find_corners_subpix = True

# input paths
base_dir = "Data"
base_dir_mono = "Mono 1"
base_dir_stereo = "Stereo 2"
images_dir1 = "cam2"
images_dir2 = "cam3"

# choose the directories to calibrate the cameras
chosen_dir_calib = os.path.join(base_dir, base_dir_stereo)
# choose the directories to calibrate the stereo system
chosen_dir_stereo_calib = os.path.join(base_dir, base_dir_stereo)

# images paths for calibration
imgs_path_calib1 = os.path.join(chosen_dir_calib, images_dir1)
imgs_path_calib2 = os.path.join(chosen_dir_calib, images_dir2)

# images to check stereo rectification
example_img = "23.png"
example_img_path1 = os.path.join(os.path.join(chosen_dir_stereo_calib, images_dir1), example_img)
example_img_path2 = os.path.join(os.path.join(chosen_dir_stereo_calib, images_dir2), example_img)

# output paths
output_json1 = "calib_params_" + images_dir1 + ".json"
output_json2 = "calib_params_" + images_dir2 + ".json"
output_json_stereo = "stereo_calib_params_" + images_dir1 + "_" + images_dir2 + ".json"

# prefix for images with detected corners
img_with_pattern = "chessboard_" 

# lists of images that succeeded and failed in the calibration process
succeeded = []
failed = []

# debug
img_num = os.path.basename(example_img).rsplit(".", 1)[0]
imgs_to_check = [img_num]
DEBUG = False

# ---------------------------------

# ---------- CALIBRATION ----------

# Function responsible for calibrating cameras and saving the calibration parameters to a file
def calibrate_cameras(chessboard_dim, single_square, images_path1, images_path2):
    print("\n===== Starting chessboard detection =====")
    # termination criteria - max number of iterations and accuracy
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    objp = np.zeros((chessboard_dim[0]*chessboard_dim[1],3), np.float32)
    # 0 ... 7, 0...10
    objp[:,:2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1,2) * single_square
    
    # arrays to store coordinates of checkerboards
    objpoints = []      # 3d point in real world space
    imgpoints_1 = []    # 2d points in image plane for camera 1
    imgpoints_2 = []    # 2d points in image plane for camera 2
    
    # load images from both directories
    paths1 = os.path.join(images_path1, "*.png")
    paths2 = os.path.join(images_path2, "*.png")
    images1 = glob.glob(paths1)
    images2 = glob.glob(paths2)
    
    counter = 0

    # iterate through the images
    for fname1, fname2 in zip(images1, images2):
        counter += 1

        # assume that the images are named in format: 1.png, 2.png, 3.png, ...
        # TODO - write a function that assures that the images names are in this format
        img_num = os.path.basename(fname1).rsplit(".", 1)[0]

        img_1 = cv.imread(fname1)
        img_2 = cv.imread(fname2)
        gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
        gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

        img_size_1 = gray_1.shape[::-1]
        img_size_2 = gray_2.shape[::-1]

        # check if the image sizes match
        if img_size_1 != img_size_2:
            print(f"Image sizes do not match: {img_size_1} and {img_size_2} - rejecting images...")
            continue

        # find the chessboard corners
        ret_1, corners_1 = cv.findChessboardCorners(gray_1, chessboard_dim, None)
        ret_2, corners_2 = cv.findChessboardCorners(gray_2, chessboard_dim, None)
    
        # if found, add object points and image points (after refining them)
        if ret_1 == True and ret_2 == True:
            objpoints.append(objp)

            # refine the corners
            if find_corners_subpix == True:
                corners_1 = cv.cornerSubPix(gray_1, corners_1, (11,11), (-1,-1), criteria)
                corners_2 = cv.cornerSubPix(gray_2, corners_2, (11,11), (-1,-1), criteria)

            # append the corners to the lists
            imgpoints_1.append(corners_1)
            imgpoints_2.append(corners_2)
    
            # draw and display the corners
            cv.drawChessboardCorners(img_1, chessboard_dim, corners_1, ret_1)
            cv.drawChessboardCorners(img_2, chessboard_dim, corners_2, ret_2)
            
            # append the image number to the list of succeeded images
            succeeded.append(img_num)

            # save example images with detected corners
            if img_num in imgs_to_check:
                cv.imwrite(img_with_pattern + str(img_num) + "_" + images_path1 + ".png", img_1)
                cv.imwrite(img_with_pattern + str(img_num) + "_" + images_path2 + ".png", img_2)       
        else:
            failed.append(img_num)

    # end of iteration through the images, print the results
    print(f"Processed {counter} images.")
    print(f"VVV Images number with successful detection: {len(succeeded)}")
    failed.sort()
    print(f"XXX Images number with failed detection: {len(failed)}: {failed}")

    # calculate internal parameters of cameras
    print("\n===== Camera calibration =====")
    ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv.calibrateCamera(objpoints, imgpoints_1, img_size_1, None, None)
    ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = cv.calibrateCamera(objpoints, imgpoints_2, img_size_2, None, None)
    print("Camera 1 reprojection RMS error: ", ret_1)
    print("Camera 2 reprojection RMS error: ", ret_2)

    mean_error_1 = calc_reprojection_error(objpoints, imgpoints_1, rvecs_1, tvecs_1, mtx_1, dist_1)
    mean_error_2 = calc_reprojection_error(objpoints, imgpoints_2, rvecs_2, tvecs_2, mtx_2, dist_2)
    print( "Camera 1 total reprojection error: {}".format(mean_error_1 / len(objpoints)) )
    print( "Camera 2 total reprojection error: {}".format(mean_error_2 / len(objpoints)) )

    save_params_to_json(objpoints, imgpoints_1, mtx_1, dist_1, rvecs_1, tvecs_1, mean_error_1, output_json1)
    save_params_to_json(objpoints, imgpoints_2, mtx_2, dist_2, rvecs_2, tvecs_2, mean_error_2, output_json2)


# TASK 1 - stereo calibrate
# Function responsible for stereo calibration
def stereo_calibrate(objpoints, imgpoints_1, imgpoints_2, img_size, mtx_1, dist_1, mtx_2, dist_2):
    print("\n===== Stereo calibration =====")
    # flags - we fix the intrinsic parameters of the cameras
    flags = cv.CALIB_FIX_INTRINSIC
    # termination criteria - max number of iterations and accuracy
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.001)

    # stereo calibration
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_1, imgpoints_2, mtx_1, dist_1, mtx_2, dist_2, img_size, criteria=criteria, flags=flags)
    print("Stereo calibration RMS error: ", ret)

    # TASK 2 - calculate baseline - the distance between two cameras
    # baseline in mm (single square size in mm)
    baseline = np.linalg.norm(T)
    # baseline in cm
    #baseline = np.round((np.linalg.norm(T)) * 0.1, 2)
    print("Baseline [mm]: ", baseline)

    fov_1 = calculate_fov(mtx1, img_size)
    fov_2 = calculate_fov(mtx2, img_size)
    print("Camera 1 FOV (degrees): ", fov_1, "Camera 2 FOV (degrees): ", fov_2)

    save_stereo_params_to_json(ret, mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov_1, fov_2)

    return mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov_1, fov_2


# TASK 3 - rectify stereo images
# Function that rectifies stereo images
def rectify_stereo_images(img1, img2, mtx1, dist1, mtx2, dist2, R, T):
    print("\n===== Rectifying stereo images =====")
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

    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    print("Original image size", image_size)

    cv.imwrite("unrectified_" + "_" + images_dir1 + ".png", img1)
    cv.imwrite("unrectified_" + "_" + images_dir2  + ".png", img2)

    # get rectification matrices, projection matrices and disparity-to-depth mapping matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, image_size, R, T, flags=flags, alpha=alpha)

    # get the rectification maps
    map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, image_size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, image_size, cv.CV_32FC1)

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    interpolation_methods = [
        (cv.INTER_NEAREST, "INTER_NEAREST"),
        (cv.INTER_CUBIC, "INTER_CUBIC"),
        (cv.INTER_AREA, "INTER_AREA"),
        (cv.INTER_LANCZOS4, "INTER_LANCZOS4"),
        (cv.INTER_LINEAR, "INTER_LINEAR")
    ]

    # print("roi1: ", roi1)
    # print("roi2: ", roi2)
    # print("map1x: ", map1x.shape)
    # print("map1y: ", map1y.shape)
    # print("map2x: ", map2x.shape)
    # print("map2y: ", map2y.shape)

    # cv.imwrite("NON_REMAPPED_" + chosen_dir_stereo_calib + "_" + images_dir1 + ".png", img1)
    # cv.imwrite("NON_REMAPPED_" + chosen_dir_stereo_calib + "_" + images_dir2  + ".png", img2)

    tmp = True

    results = []
    # remap the images
    for method, name in interpolation_methods:
        # start timer
        start_time = cv.getTickCount()
        img1_rectified = cv.remap(img1, map1x, map1y, method)
        img2_rectified = cv.remap(img2, map2x, map2y, method)
        # time in seconds
        elapsed_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
        print(f"Remap interpolation method: {name} | Time elapsed: {elapsed_time:.4f} seconds")
        results.append((img1_rectified, img2_rectified, name))

        # crop the images
        if crop_img == True:

            x_left_up = min(x1, x2)
            y_left_up = min(y1, y2)

            x_right_down = max(x1 + w1, x2 + w2)
            y_right_down = max(y1 + h1, y2 + h2)

            img1_rectified = img1_rectified[y_left_up : y_right_down, x_left_up : x_right_down]
            img2_rectified = img2_rectified[y_left_up : y_right_down, x_left_up : x_right_down]

            roi1 = (x1 - x_left_up, y1 - y_left_up, w1, h1)
            roi2 = (x2 - x_left_up, y2 - y_left_up, w2, h2)

            if tmp == True:
                tmp = False
                print ("ROI1: ", roi1, "ROI2: ", roi2) 


        # save rectified imgs
        cv.imwrite("REMAPPED_" + "_" + images_dir1 + "_" + name + ".png", img1_rectified)
        cv.imwrite("REMAPPED_" + "_" + images_dir2  + "_" + name + ".png", img2_rectified)

        # TASK 5 - draw ROI and epipolar lines on the rectified images
        draw_roi_and_epilines(img1_rectified, img2_rectified, roi1, roi2)

        # TASK 4 - display before and after rectification
        if DEBUG:
            display_before_after_rectification(img1, img2, img1_rectified, img2_rectified)

    return img1_rectified, img2_rectified


# Function responsible for displaying images before and after rectification
def display_before_after_rectification(img1, img2, img1_rectified, img2_rectified):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        axs[0, 0].set_title('Before rectification - Camera 1')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
        axs[0, 1].set_title('Before rectification - Camera 2')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(cv.cvtColor(img1_rectified, cv.COLOR_BGR2RGB))
        axs[1, 0].set_title('After rectification - Camera 1')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(cv.cvtColor(img2_rectified, cv.COLOR_BGR2RGB))
        axs[1, 1].set_title('After rectification - Camera 2')
        axs[1, 1].axis('off')
        plt.show()


# TASK 5
# Function responsible for drawing the region of interest and epipolar lines on the rectified images
def draw_roi_and_epilines(img1, img2, roi1, roi2):
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    line_thickness_roi = 3
    line_thickness_lines = 2
    lines_count = 20

    # draw the ROIs
    color = (0, 255, 0)
    cv.rectangle(img1_copy, (x1, y1), (x1 + w1, y1 + h1), color, line_thickness_roi)
    cv.rectangle(img2_copy, (x2, y2), (x2 + w2, y2 + h2), color, line_thickness_roi)

    # concatenate the images
    rectified_pair = np.hstack((img1_copy, img2_copy))

    # draw epipolar lines
    new_height, new_width = rectified_pair.shape[:2]
    color = (0, 0, 255)
    y_coords = np.linspace(0, new_height - 1, lines_count).astype(int)
    for y in y_coords:
        cv.line(rectified_pair, (0, y), (new_width - 1, y), color, line_thickness_lines)

    # TASK 6 - export images with ROIs and epipolar lines to a PNG file
    # save the image with ROIs and epipolar lines drawn
    cv.imwrite("rectified_with_ROI.png", rectified_pair)


# Function that calculates the field of view of the camera
def calculate_fov(camera_matrix, img_size):
    
    # camera_matrix:
    # | fx 0 cx |
    # | 0 fy cy |
    # | 0  0  1 |
    # fx - focal length in x direction
    # fy - focal length in y direction
    # cx - principal point in x direction
    # cy - principal point in y direction

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    width, height = img_size

    fov_horizontal = 2 * np.arctan2(width, (2 * fx))
    fov_horizontal_deg = fov_horizontal * 180 / np.pi

    fov_vertical = 2 * np.arctan2(height, (2 * fy))
    fov_vertical_deg = fov_vertical * 180 / np.pi

    return fov_horizontal_deg, fov_vertical_deg

# ------------------------------------------------

# ---------- JSON FUNCTIONS ----------

# Function that saves the calibration parameters to a file
def save_params_to_json(objpoints, imgpoints, mtx, dist, rvecs, tvecs, mean_error, path):
    calibration_data = {
        "object_points": [objp.tolist() for objp in objpoints],
        "image_points": [imgp.tolist() for imgp in imgpoints],
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs],
        "reprojection_error": mean_error / len(objpoints)
    }

    with open(path, 'w') as f:
        json.dump(calibration_data, f, indent=4)   
    print("Calibration parameters saved to: " + path)

# Function that reads the calibration parameters from a file
def read_params_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        objpoints = [np.array(objp, dtype=np.float32) for objp in data["object_points"]]
        imgpoints = [np.array(imgp, dtype=np.float32) for imgp in data["image_points"]]
        mtx = np.array(data["camera_matrix"])
        dist = np.array(data["distortion_coefficients"])
        rvecs = [np.array(rvec) for rvec in data["rotation_vectors"]]
        tvecs = [np.array(tvec) for tvec in data["translation_vectors"]]
        mean_error = data["reprojection_error"]
        print("Calibration parameters loaded from: " + path)
    return objpoints, imgpoints, mtx, dist, rvecs, tvecs, mean_error


# Function that saves the stereo calibration parameters to a file
def save_stereo_params_to_json(ret, mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov1, fov2):
    stereo_calibration_data = {
        "rms_error": ret,
        "left_camera_matrix": mtx1.tolist(),
        "left_distortion_coefficients": dist1.tolist(),
        "right_camera_matrix": mtx2.tolist(),
        "right_distortion_coefficients": dist2.tolist(),
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "baseline": baseline,
        "left_camera_fov": fov1,
        "right_camera_fov": fov2
    }
    with open(output_json_stereo, 'w') as f:
        json.dump(stereo_calibration_data, f, indent=4)   
    print("Stereo calibration parameters saved to: " + output_json_stereo)

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

# ------------------------------------------------

# ---------- OTHER FUNCTIONS ----------

# Function that checks if the number of images in both directories is the same
def check_images_count(images_path1, images_path2):
    paths1 = os.path.join(images_path1, "*.png")
    paths2 = os.path.join(images_path2, "*.png")
    images1 = glob.glob(paths1)
    images2 = glob.glob(paths2)

    if len(images1) == len(images2):
        print(f"\nImages count in both directories is the same: {len(images1)}")
        return True
    else:
        print(f"\nImages count in directories is different: {len(images1)} and {len(images2)}  - exiting...")
        return False
    

# Function that calculates the reprojection error based on the calibration parameters
# It represents how good realspace points are projected onto the image plane using the calibration parameters
def calc_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_new, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints_new, cv.NORM_L2) / len(imgpoints_new)
        mean_error += error
    return mean_error

# ------------------------------------------------

# ----- MAIN -----

print("\n===== STARTING PROGRAM =====")  

# check if the number of images in both directories is the same - if not, exit the program
if check_images_count(imgs_path_calib1, imgs_path_calib2) == False:
    exit()

# if the number of images in both directories is the the same, we can proceed with the calibration
if CAMERAS_CALIBRATE:
    calibrate_cameras(chessboard_dim, single_square, imgs_path_calib1, imgs_path_calib2)

if STEREO_CALIBRATE == False:
    exit()

# read the calibration parameters from the files
objpoints1, imgpoints1, mtx1, dist1, rvecs1, tvecs1, mean_error1 = read_params_from_json(output_json1)
objpoints2, imgpoints2, mtx2, dist2, rvecs2, tvecs2, mean_error2 = read_params_from_json(output_json2)

error1 = calc_reprojection_error(objpoints1, imgpoints1, rvecs1, tvecs1, mtx1, dist1)
error2 = calc_reprojection_error(objpoints2, imgpoints2, rvecs2, tvecs2, mtx2, dist2)
print("Camera 1 total reprojection error: ", error1 / len(objpoints1))
print("Camera 2 total reprojection error: ", error2 / len(objpoints2))

# read the stereo calibration parameters from the file
if READ_STEREO_CALIB_PARAMS == True:
    print("\n===== Reading stereo calibration parameters from file =====")
    mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov1, fov2 = read_stereo_params_from_json(output_json_stereo)
else:
    # image size
    sample_img = cv.imread(example_img_path1)
    gray = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    # stereo calibration
    mtx1, dist1, mtx2, dist2, R, T, E, F, baseline, fov1, fov2 = stereo_calibrate(objpoints1, imgpoints1, imgpoints2, img_size, mtx1, dist1, mtx2, dist2)

# rectify stereo images
img1 = cv.imread(example_img_path1)
img2 = cv.imread(example_img_path2)
img1_rectified, img2_rectified = rectify_stereo_images(img1, img2, mtx1, dist1, mtx2, dist2, R, T)
