# Author: Wiktoria Marczyk
# Lab 4: Depth Map and Point Cloud Generation

import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- PARAMETERS ----------

ref_disp_map_Z1_path = "Data/Z1/disp0.pfm"
img_left_Z1_path = "Data/Z1/im0.png"
img_right_Z1_path = "Data/Z1/im1.png"
Z1_calib_params_path = "Data/Z1/calib.txt"

ref_depth_map_Z4_path = "Data/Z4/depth.png"
img_left_Z4_path = "Data/Z4/left.png"

ref_disp_Z1_output = "ref_disp_map_Z1.png"
disp_stereoSGBM_output = "disp_map_stereoSGBM_Z1.png"
depth_output_1 = "depth_map_from_disp_Z1.png"
depth_output_2 = "depth_map_stereoSGBM_Z1.png"
depth_map_24bit_output = "depth_map_24bit_Z1.png"

disp_output = "disp_map_from_depth_Z4.png"

ply_file_name = "point_cloud.ply"

# ---------------------------------

# ---------- FUNCTIONS ----------

def read_calib_params_from_txt(path):
    """
    Reads calibration parameters from a txt file.

    Args:
        path (str): Path to the calibration parameters in txt file.

    Returns:
        tuple: Contains the following parameters:
            - mtx1 (np.array): Camera matrix for the first camera. [fx 0 cx; 0 fy cy; 0 0 1]
            - mtx2 (np.array): Camera matrix for the second camera.
            - doffs (float): Disparity offset - offset in pixels between the cameras on the x-axis abs(cx1 - cx2).
            - width (int): Image width.
            - height (int): Image height.
            - baseline (float): Distance between cameras in the same units as focal length (usually in mm).
            - ndisp (int): Number of disparities - maximum disparity minus minimum disparity.
            - isint (bool): Whether disparity is integer (1) or fractional (0).
            - vmin (float): Minimum disparity value.
            - vmax (float): Maximum disparity value.
            - dyavg (float): Average vertical disparity between the cameras in pixelsa (0 - identical).
            - dymax (float): Maximum vertical disparity between the cameras in pixels (0 - identical).
    """
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            params[key.strip()] = value.strip()
    
    # Parse camera matrices and other parameters
    try:
        def parse_matrix(value):
            """Parses a string matrix in the format [a b c; d e f; g h i]."""
            value = value.strip("[]")
            rows = value.split(";")
            return np.array([[float(num) for num in row.split()] for row in rows])
        
        mtx1 = parse_matrix(params["cam0"])
        mtx2 = parse_matrix(params["cam1"])
        doffs = float(params["doffs"])
        baseline = float(params["baseline"])
        width = int(params["width"])
        height = int(params["height"])
        ndisp = int(params["ndisp"])
        isint = bool(int(params["isint"]))
        vmin = float(params["vmin"])
        vmax = float(params["vmax"])
        dyavg = float(params["dyavg"])
        dymax = float(params["dymax"])

    except KeyError as e:
        raise ValueError(f"Missing key in calibration file: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing calibration file: {e}")

    print("Calibration parameters loaded from: ", path)
    return mtx1, mtx2, doffs, width, height, baseline, ndisp, isint, vmin, vmax, dyavg, dymax


def read_pfm(file_path):
    """
    Reads a PFM file and returns the data as a NumPy array.

    Args:
        file_path (str): Path to the PFM file.

    Returns:
        np.ndarray: Array containing the PFM data.
    """
    with open(file_path, 'rb') as f:
        # Read the header to check format
        header = f.readline().decode('utf-8').strip()
        if header != 'Pf':
            raise ValueError("Unsupported PFM format. Only grayscale 'Pf' is supported.")
        
        # Read the dimensions
        dimensions = f.readline().decode('utf-8').strip()
        try:
            width, height = map(int, dimensions.split())
        except ValueError:
            raise ValueError("Invalid dimensions in PFM file.")
        
        # Read the scale
        scale = float(f.readline().decode('utf-8').strip())
        if scale == 0:
            raise ValueError("Scale value cannot be zero.")
        
        # Determine endianness
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # Read the pixel data
        data = np.fromfile(f, dtype=endian + 'f')
        if data.size != width * height:
            raise ValueError("Mismatch between file size and header dimensions.")
        
        # Reshape and flip vertically (PFM stores data bottom to top)
        data = np.reshape(data, (height, width))
        data = np.flipud(data)  # Flip the data along the vertical axis

    return data
    

def disparity_to_depth(disp_map, focal_length, baseline):
    """Converts a disparity map to a depth map."""
    print("\n===== Converting disparity map to a depth map =====")
    print("Z = f*B / d")
    # Z - depth, f - focal length, B - baseline, d - disparity
    disp_map = disp_map.astype(np.float32)
    # Handle division by zero
    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / disp_map
    # Set invalid disparities to 0 depth
    depth_map[disp_map <= 0] = 0
    
    return depth_map


def normalize_to_8bit_map(depth_map):
    """Normalizes the depth map to 8-bit range (0-255)."""
    print("\n===== Normalizing depth map to 8-bit range =====")
    normalized_depth = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    normalized_depth = np.uint8(normalized_depth)
    return normalized_depth


def stereoSGBM(img_left_path, img_right_path, min_disparity=0, block_size=5, num_disparities=256):
    """
    Calculates the disparity map using the StereoSGBM (Semi-Global Block Matching) algorithm.
    :param img_left_path: Path to the left image.
    :param img_right_path: Path to the right image.
    :param ref_img: Reference disparity map.
    :param min_disparity: Minimum possible disparity value. Normally, it is 0.
    :param block_size: Size of the block used for matching. It must be an odd number >=1. Normally, it should be somewhere in the 3...11 range.
    :param num_disparities: Maximum disparity minus minimum disparity. The value is always greater than zero. Must be divisible by 16.
    """
    print("\n===== Computing disparity map using StereoSGBM algorithm =====")

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
    channels = 3
    p1 = 8*channels*block_size*block_size  # Penalty on the disparity change by plus or minus 1 between neighbor pixels
    p2 = 32*channels*block_size*block_size # Penalty on the disparity change by more than 1 between neighbor pixels 
    disp12_max_diff = 1             # Max disparity difference between left-right checks. Set it to a non-positive value to disable the check.
    uniqueness_ratio = 10           # Percentage of uniqueness for the disparity match. Normally, a value within the 5-15 range is good enough.
    speckle_window_size = 100       # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it in the 50-200 range.
    speckle_range = 32              # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough
    
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
    disparity = stereo.compute(left_img, right_img)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity


# NOTE: Assuming that max depth is 1000 meters, for 8-bit depth map resolution is around 4 meters (1000/256)
def depth_map_to_norm_24bit_rgb(depth_map):
    """Converts depth map to a 24-bit RGB image."""
    print("\n===== Converting depth map to a 24-bit RGB image =====")

    # Normalize depth values to 24-bit integer range (0-16777215) - 16777215 = 2^24 - 1
    depth_scaled = cv.normalize(depth_map, None, 0, 16777215, cv.NORM_MINMAX)
    depth_scaled = depth_scaled.astype(np.uint32)

    # Split 24-bit depth values into RGB channels
    r = (depth_scaled & 0xFF).astype(np.uint8)
    g = ((depth_scaled >> 8) & 0xFF).astype(np.uint8)
    b = ((depth_scaled >> 16) & 0xFF).astype(np.uint8)

    # Combine into an RGB image
    rgb_image = np.stack((r, g, b), axis=-1)
    return rgb_image


def normalized_24bit_to_8bit(rgb_image):
    """Converts a 24-bit rgb image to an 8-bit depth map."""
    print("\n===== Converting 24-bit RGB image to 8-bit depth map =====")

    # Extract RGB channels
    r = rgb_image[:, :, 0].astype(np.uint32)
    g = rgb_image[:, :, 1].astype(np.uint32)
    b = rgb_image[:, :, 2].astype(np.uint32)

    # Decode image [0, 16777215] to normalized depth map [0, 1]
    normalized_depth = (r + (g << 8) + (b << 16)) / 16777215.0

    # Convert normalized depth map to grayscale [0, 255]
    depth_map  = (normalized_depth * 255)

    return depth_map 


# 24-bit depth map to 8-bit disparity map
def depth_to_disparity(depth_map, baseline, h_fov, min_depth, max_depth):
    """Converts a disparity map to a depth map."""
    print("\n===== Converting depth map to disparity map =====")
    print("d = f*B / Z")

    # Decode depth from RGB channels
    r = depth_map[:, :, 2].astype(np.uint32)
    g = depth_map[:, :, 1].astype(np.uint32)
    b = depth_map[:, :, 0].astype(np.uint32)

    # Depth in meters
    depth = ((r + (g << 8) + (b << 16)) / 16777215.0) * (max_depth - min_depth) + min_depth

    # Calculate focal length in pixels from horizontal field of view and img width
    width = depth.shape[1]
    f = calculate_focal_length(width, h_fov)

    # Calculate disparity map
    # Z = f*B / d
    # d = f*B / Z
    # Z - depth, f - focal length, B - baseline, d - disparity
    disparity = (f * baseline) / depth

    # Normalize disparity map to range [0, 255]
    disparity_norm = normalize_to_8bit_map(disparity)

    return disparity_norm


def calculate_focal_length(width, h_fov=60.0):
    """Calculate focal length from image width and horizontal field of view."""
    # angles to radians
    h_fov = math.radians(h_fov)
    x = math.tan(h_fov / 2)
    f = width / (2 * x)
    return f


def compare_maps(map1, map2, title1, title2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(map1, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(map2, cmap='gray')
    plt.colorbar()

    plt.show()


def depth_to_point_cloud(depth_map, color_image, clip_distance, max_depth, baseline, h_fov, ply_file_name):
    """Convert depth map and color image to a 3D point cloud."""
    print("\n===== Generating point cloud from depth map =====")

    height = depth_map.shape[0]
    width = depth_map.shape[1]
    
    focal_length = calculate_focal_length(width, h_fov)

    Q = np.float32([[1, 0, 0, -0.5*width],
                    [0,-1, 0, 0.5*height],           # turn points 180 deg around x-axis,
                    [0, 0, 0, -focal_length],   # so that y-axis looks up
                    [0, 0, 1, 0]])
    
    # Compute disparity map from depth map
    disparity = depth_to_disparity(depth_map, baseline=baseline, h_fov=h_fov, min_depth=0, max_depth=max_depth)
    # Compute 3D points from disparity map
    points = cv.reprojectImageTo3D(disparity, Q)

    clip_distance_sq = clip_distance**2
    out_points = []
    out_colors = []
    index = 0

    # Filter points that are too far away (clip distance)
    for line in points:
        for point in line:
            x, y, z = point
            if x**2 + y**2 + z**2 < clip_distance_sq:
                out_points.append(point)
                color = color_image[index // width, index % width]
                # swap red and blue channels
                color = [color[2], color[1], color[0]]
                out_colors.append(color)
            index += 1

    # Save point cloud to a PLY file
    save_to_ply(ply_file_name, out_points, out_colors)


def save_to_ply(filename, points, colors):
    """Save points and colors to a PLY file."""
    print(f"\n===== Saving point cloud to {filename} =====")
    with open(filename, 'w') as f:
        # Header PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


# ---------------------------------

# ----- MAIN -----
print("\n===== STARTING PROGRAM =====")  

# Read the reference disparity map in pfm format
ref_disp_map = read_pfm(ref_disp_map_Z1_path)
# Save it as a png file
cv.imwrite(ref_disp_Z1_output, ref_disp_map)
# Read the reference disparity map in png format
ref_disp_map = cv.imread(ref_disp_Z1_output, cv.IMREAD_GRAYSCALE)

# ----- TASK 1 - Convert the disparity map to a depth map -----
mtx1, mtx2, doffs, width, height, baseline, ndisp, isint, vmin, vmax, dyavg, dymax = read_calib_params_from_txt(Z1_calib_params_path)
focal_length = mtx1[0, 0]
depth_map = disparity_to_depth(ref_disp_map, focal_length, baseline)
norm_depth_map = normalize_to_8bit_map(depth_map)
compare_maps(ref_disp_map, norm_depth_map, "Ref disparity map", "Depth map from disparity map")
cv.imwrite(depth_output_1, norm_depth_map)


# ----- TASK 2 - Convert the disparity map computed with StereoSGBM to a depth map -----
min_disparity = int(vmin)
num_disparities = math.ceil((vmax - vmin) / 16) * 16
# Compute disparity map using StereoSGBM
disp_map_stereoSGBM = stereoSGBM(img_left_Z1_path, img_right_Z1_path, min_disparity, 5, num_disparities) # min=16 size=3 num=112-16

# Crop disparity maps to remove the black areas
ref_disp_map = ref_disp_map[:, int(ndisp):]
disp_map_stereoSGBM = disp_map_stereoSGBM[:, int(ndisp):]

cv.imwrite(disp_stereoSGBM_output, disp_map_stereoSGBM)
compare_maps(ref_disp_map, disp_map_stereoSGBM, "Ref disparity map", "Disparity map from StereoSGBM")

# Convert the disparity map (StereoSGBM) to depth map
depth_map_2 = disparity_to_depth(disp_map_stereoSGBM, focal_length, baseline)
norm_depth_map_2 = normalize_to_8bit_map(depth_map_2)
cv.imwrite(depth_output_2, norm_depth_map_2)
compare_maps(norm_depth_map, norm_depth_map_2, "Ref depth map ", "Depth map from StereoSGBM")


# ----- TASK 3 - Convert the depth map to 24-bit image-----
rgb_depth_map = depth_map_to_norm_24bit_rgb(depth_map)
cv.imwrite(depth_map_24bit_output, rgb_depth_map)
compare_maps(norm_depth_map, rgb_depth_map, "Depth Map", "24-bit RGB Image")

# Revert 24-bit RGB image to 8-bit depth map
depth_map_from_rgb = normalized_24bit_to_8bit(rgb_depth_map)
compare_maps(norm_depth_map, depth_map_from_rgb, "Depth Map", "Depth Map from 24-bit RGB Image")


# ----- TASK 4 - Compute depth map from disparity map-----
depth_map_3 = cv.imread(ref_depth_map_Z4_path)
disp_map_from_depth = depth_to_disparity(depth_map_3, baseline=0.1, h_fov=60, min_depth=0, max_depth=1000)
cv.imwrite(disp_output, disp_map_from_depth)
compare_maps(depth_map_3, disp_map_from_depth, "Depth Map", "Disparity map from depth map")


# ----- TASK 5 - Convert depth map to point cloud -----
color_image = cv.imread(img_left_Z4_path)
depth_to_point_cloud(depth_map_3, color_image, clip_distance=50.0, max_depth=1000, baseline=0.1, h_fov=60, ply_file_name=ply_file_name)

# ----------------------------------------