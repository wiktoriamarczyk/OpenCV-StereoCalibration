# Author: Wiktoria Marczyk
# Lab 1 - Camera Calibration

import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json
 
 # ---------- PARAMETERS ----------

# define the dimensions of the chessboard (number of inner corners per row and column)
chessboard_dim = (7, 10)

# define the size of a single chessboard square in mm
single_square = 28.67

# define method to find corners, True for cornerSubPix, False for findChessboardCorners
find_corners_subpix = True 

# paths
images_dir = "Data/Mono 1"
img_to_undistort = "1.png"
# output images
img_with_pattern = "chessboard_"
distort_img = "distorted.png"
undistort_img_1= "undistorted.png"
undistort_img_2 = "undistorted_remap.png"

# debug
imgs_to_check = [1]
succeeded = []
failed = []
DEBUG = True

# ---------------------------------

# Function that returns the name of the method used to find corners
def which_corners_finding_method():
    if find_corners_subpix:
        return "cornerSubPix"
    else:
        return "findChessboardCorners"

# file in which the calibration parameters will be saved
output_json = "calibration_parameters_" + which_corners_finding_method() + ".json"

# Function responsible for calibrating the camera and saving the calibration parameters to a file
def calibrate_camera(chessboard_dim, single_square, images_path):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_dim[0]*chessboard_dim[1],3), np.float32)
    # 0 ... 7, 0...10
    objp[:,:2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1,2) * single_square
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    counter = 0
    print("\n=====Starting chessboard detection - method of finding corners: " + which_corners_finding_method()  + "=====")
    
    paths = os.path.join(images_path, "*.png")
    images = glob.glob(paths)
    
    for fname in images:
        img_num = os.path.basename(fname).rsplit(".", 1)[0]
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        corners = None
        counter += 1

        # TASK 1 - Find the chessboard corners
        ret, corners_1 = cv.findChessboardCorners(gray, chessboard_dim, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # Find more precise corner coordinates
            corners_2 = cv.cornerSubPix(gray, corners_1, (11,11), (-1,-1), criteria)

            # Use corners based on the method chosen
            if find_corners_subpix:
                corners = corners_2
            else:
                corners = corners_1

            imgpoints.append(corners)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboard_dim, corners, ret)
                
            succeeded.append(img_num)

            if counter in imgs_to_check:
                cv.imwrite(img_with_pattern + str(counter) + "_" + which_corners_finding_method() + ".png", img)
                cv.imwrite(img_with_pattern + str(counter) + "_" + which_corners_finding_method() + "_gray.png", gray)
        
        else:
            failed.append(img_num)

    succeeded.sort()
    failed.sort()
    print(f"Processed {counter} images.")
    print(f"VVV Images numbers with successful detection ({len(succeeded)} images)")
    print(f"XXX Images numbers with failed detection ({len(failed)} images): {failed}")


    # TASK 2 - Calculate internal parameters
    print("\n=====Calibrating camera=====")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # TASK 3 - Calculate the reprojection error based on the calibration parameters
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Total error: {}".format(mean_error/len(objpoints)) )

    save_parameters_to_json(objpoints, mtx, dist, rvecs, tvecs, mean_error)
   

# TASK 4 - Save the calibration parameters to a file
# Save the calibration parameters to a JSON file
def save_parameters_to_json(objpoints, mtx, dist, rvecs, tvecs, mean_error):
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs],
        "reprojection_error": mean_error / len(objpoints)
    }

    with open(output_json, 'w') as f:
        json.dump(calibration_data, f, indent=4)   
    print("Parameters saved to: " + output_json)


# Read the calibration parameters from a file
def read_parameters_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        mtx = np.array(data["camera_matrix"])
        dist = np.array(data["distortion_coefficients"])
        rvecs = np.array(data["rotation_vectors"])
        tvecs = np.array(data["translation_vectors"])
        mean_error = data["reprojection_error"]
        print("Parameters loaded from: " + path)
    return mtx, dist, rvecs, tvecs, mean_error


# TASK 5 - Remove distortion from an image - undistort
# Remove distortion from an image using cv.undistort
def remove_distortion_using_undistort(image_path, mtx, dist):
    print("\n=====Removing distortion from image - undistort=====")
    # load image
    img = cv.imread(image_path)

    h, w = img.shape[:2]
    print(f"Original - W: {w} H: {h}")
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv.imwrite(undistort_img_1, dst)

    if DEBUG:
        print(f"After - W: {w} H: {h}")
        cv.imshow('Distorted image', img)
        cv.waitKey(0)
        cv.imshow('Undistorted image', dst)
        cv.waitKey(0)


# TASK 6 - Remove distortion from an image - initUndistortRectifyMap + remap
# Remove distortion from an image using cv.initUndistortRectifyMap and cv.remap
def remove_distortion_using_remap(image_path, mtx, dist):
    print("\n=====Removing distortion from image - remap=====")
    # load image
    img = cv.imread(image_path)
    
    h, w = img.shape[:2]
    print(f"Original - W: {w} H: {h}")
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv.imwrite(undistort_img_2, dst)

    if DEBUG:
        print(f"After - W: {w} H: {h}")
        cv.imshow('Distorted image', img)
        cv.waitKey(0)
        cv.imshow('Undistorted image', dst)
        cv.waitKey(0)

# ---------------------------------

# ----- MAIN -----

calibrate_camera(chessboard_dim, single_square, images_dir)
mtx, dist, rvecs, tvecs, mean_error = read_parameters_from_json(output_json)
img_to_undistort_path = os.path.join(images_dir, img_to_undistort)
cv.imwrite(distort_img, cv.imread(img_to_undistort_path))
remove_distortion_using_undistort(img_to_undistort_path, mtx, dist)
remove_distortion_using_remap(img_to_undistort_path, mtx, dist)

# ---------------------------------