# Author: Wiktoria Marczyk
# Lab 5: Optical Flow

import math
import time
import numpy as np
import cv2 as cv

# ---------- PARAMETERS ----------

video_path = "Data/video.mp4"
frame_change_in_ms = 30
motion_threshold = 0.5
min_contour_area = 500
min_color_ratio = 0.5

# ---------- FUNCTIONS ----------

def sparse_optical_flow(video_path):
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100,3))

    # Initial setup
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        return

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    while(1):
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading a frame.")
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)    
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a,b), 5, color[i].tolist(), -1)

        # Display the result
        img = cv.add(frame, mask)
        cv.imshow('Sparse optical flow', img)

        # Break on ESC key
        k = cv.waitKey(frame_change_in_ms) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    cap.release()
    cv.destroyAllWindows()


def draw_bbox_and_dir(flow, frame, x, y, w ,h):
    
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Calculate the direction and speed of the object
    center_x = x + w // 2
    center_y = y + h // 2
    dx = np.mean(flow[y:y+h, x:x+w, 0])
    dy = np.mean(flow[y:y+h, x:x+w, 1])
    speed = np.sqrt(dx**2 + dy**2)
    angle = math.degrees(np.arctan2(dy, dx))

    # Display speed of the object
    offset_from_bbox = 20
    cv.putText(frame, f"Speed: {speed:.2f} px/s", (x, y+h+offset_from_bbox), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw arrow indicating the direction of the object
    arrow_length = speed * 15
    arrow_end_x = int(center_x + arrow_length * np.cos(np.radians(angle)))
    arrow_end_y = int(center_y + arrow_length * np.sin(np.radians(angle)))        
    cv.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y), (0, 0, 255), 2, tipLength=0.05)


def dense_optical_flow(video_path, detect_moving_objects=False, lower_color=None, upper_color=None):
    """
    Calculate dense optical flow using Farneback method.
    Optionally, detect moving objects using the magnitude of the optical flow.
    Optionally, detect objects of specific color.
    Parameters:
    - video_path: path to the video file. If None, the webcam will be used.
    - detect_moving_objects: if True, bounding boxes will be drawn around moving objects.
    - lower_color: lower bound of the color range for detecting specific objects. If None, all moving objects will be detected.
    - upper_color: upper bound of the color range for detecting specific objects. If None, all moving objects will be detected.
    """

    # If the image path is not provided, use the webcam
    if video_path is not None:
        cap = cv.VideoCapture(video_path)
    else:
        cam_index = 1
        cap = cv.VideoCapture(cam_index)

    if not cap.isOpened():
        print("Error: Could not open video file or webcam.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        return
    
    # display size of the video frame
    print(f"Frame size: {int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")

    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    frame_change_times = []

    while(1):
        start_time = time.time()

        ret, frame2 = cap.read()
        if not ret:
            print("End of video or error reading a frame.")
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Detect objects of specific color
        if lower_color is not None and upper_color is not None:
            # Convert the frame to HSV color space
            hsv_frame = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_frame, lower_color, upper_color)
            bgr = cv.bitwise_and(bgr, bgr, mask=mask)

        # Detect moving objects using the magnitude of the optical flow
        if detect_moving_objects:
            _, thresh = cv.threshold(mag, 2, 255, cv.THRESH_BINARY)
            thresh = np.uint8(thresh)

            # Find contours of moving objects
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv.contourArea(contour) > min_contour_area:
                    # Draw bounding box for each moving object
                    x, y, w, h = cv.boundingRect(contour)

                    # If color range is provided, check if the object is of that color
                    if lower_color is None and upper_color is None:
                        draw_bbox_and_dir(flow, frame2, x, y, w, h)
                    else:
                        roi_mask = mask[y:y+h, x:x+w]
                        specific_color_pixels = np.sum(roi_mask)
                        total_pixels = w * h
                        # If specified color pixels are more than min_color_ratio of the total pixels, draw the bounding box
                        if specific_color_pixels / total_pixels > min_color_ratio:
                            draw_bbox_and_dir(flow, frame2, x, y, w, h)

        # Calculate the time taken to process the frame
        end_time = time.time()
        frame_time = end_time - start_time
        frame_change_times.append(frame_time)
            
        # Display the result
        cv.imshow('Dense optical flow', bgr)
        if detect_moving_objects:
            cv.imshow('Detected Objects', frame2)
        
        k = cv.waitKey(frame_change_in_ms) & 0xff
        # Break on ESC key
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)

        prvs = next

    # Calculate the average time taken to process a frame
    avg_frame_time = np.mean(frame_change_times)
    print(f"Average time taken to process a frame: {avg_frame_time:.4f} s")

    cap.release()
    cv.destroyAllWindows()


# ---------- MAIN ----------

# Task 1 - Sparse Optical Flow
sparse_optical_flow(video_path)

# Task 2 - Dense Optical Flow
dense_optical_flow(video_path)

# Task 3 - Detect moving objects using dense optical flow
dense_optical_flow(video_path, detect_moving_objects=True)

# Task 4 - Detect moving objects of specific color
# Detect yellow objects
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
dense_optical_flow(None, True, lower_yellow, upper_yellow)