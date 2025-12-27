import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import copy
import csv
import tifffile
import os
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import morphology
from skimage.morphology import skeletonize_3d, medial_axis
from skimage.measure import label, regionprops
from scipy.interpolate import make_interp_spline
from skimage import measure
from skimage import data, util
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from math import atan
import statistics
from mpl_toolkits.mplot3d import Axes3D
import time
from PIL import Image
from collections import Counter
import tifffile as tiff

# =========================================== Functions =========================================================

def calculate_integral_image(image):
    integral_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if x==0 and y==0 and z==0:
                    integral_image[x, y, z] = image[x, y, z]
                elif x>0 and y==0 and z==0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x-1, y, z]
                elif x == 0 and y>0 and z==0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x, y-1, z]
                elif x==0 and y==0 and z>0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x, y, z-1]
                elif x>0 and y>0 and z==0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x, y-1, z]+integral_image[x-1, y, z]-integral_image[x-1, y-1, z]
                elif x>0 and y==0 and z>0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x, y, z-1]+integral_image[x-1, y, z]-integral_image[x-1, y, z-1]
                elif x==0 and y>0 and z>0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x, y, z-1]+integral_image[x, y-1, z]-integral_image[x, y-1, z-1]
                elif x>0 and y>0 and z>0:
                    integral_image[x, y, z] = image[x, y, z]+integral_image[x-1, y-1, z-1]+integral_image[x, y, z-1]+integral_image[x, y-1, z]+integral_image[x-1, y, z]-integral_image[x-1, y-1, z]-integral_image[x-1, y, z-1]-integral_image[x, y-1, z-1]

    return integral_image

def calculate_threshold_average(integral_image, step): # step have to be posotive odd
    threshold_image = np.zeros_like(integral_image)
    p = round((step-1)/2)
    for x in range(integral_image.shape[0]):
        for y in range(integral_image.shape[1]):
            for z in range(integral_image.shape[2]):
                minx = max(0, x-p)
                maxx = min(integral_image.shape[0]-1, x+p)
                miny = max(0, y-p)
                maxy = min(integral_image.shape[1]-1, y+p)
                minz = max(0, z-p)
                maxz = min(integral_image.shape[2]-1, z+p)
                count = (maxx-minx+1)*(maxy-miny+1)*(maxz-minz+1)
                if minx==0 and miny==0 and minz==0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz])/count
                elif minx>0 and miny==0 and minz==0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx-1, maxy, minz])/count
                elif minx == 0 and miny>0 and minz==0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, maxy-1, minz])/count
                elif minx==0 and miny==0 and minz>0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, maxy, minz-1])/count
                elif minx>0 and miny>0 and minz==0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, miny-1, maxz]-integral_image[minx-1, maxy, maxz]+integral_image[minx-1, miny-1, maxz])/count
                elif minx>0 and miny==0 and minz>0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, maxy, minz-1]-integral_image[minx-1, maxy, maxz]+integral_image[minx-1, maxy, minz-1])/count
                elif minx==0 and miny>0 and minz>0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, maxy, minz-1]-integral_image[maxx, miny-1, maxz]+integral_image[maxx, miny-1, minz-1])/count
                elif minx>0 and miny>0 and minz>0:
                    threshold_image[x, y, z] = (integral_image[maxx, maxy, maxz]-integral_image[maxx, maxy, minz-1]-integral_image[maxx, miny-1, maxz]-integral_image[minx-1, maxy, maxz]+integral_image[minx-1, miny-1, maxz]+integral_image[minx-1, maxy, minz-1]+integral_image[maxx, miny-1, minz-1]-integral_image[minx-1, miny-1, minz-1])/count
    return threshold_image

# This function is for transforming image into int16 figure
def int16nize(image):# For goabal thresholding
    int16_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if image[x, y, z] < -32767:
                    int16_image [x, y, z] = -32767
                elif image[x, y, z] > 32767:
                    int16_image [x, y, z] = 32767
                else:
                    int16_image [x, y, z] = image [x, y, z]
    return int16_image

def global_threshold (image, T):
    global_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if image[x, y, z] >= T: # 32767
                    global_image [x, y, z] = 1
                else:
                    global_image [x, y, z] = 0
    return global_image

def global_threshold_origin (image, T):
    global_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if image[x, y, z] >= T:
                    global_image [x, y, z] = image[x, y, z]
                else:
                    global_image [x, y, z] = 0
    return global_image

# Binarize the image based on a variant scale t
def binarize (image, threshold_image, t, min):
    masked_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                if image[x, y, z]>min and image[x, y, z]>(threshold_image[x, y, z]*t):
                    masked_image[x, y, z] = 1
                else:
                    masked_image[x, y, z] = 0
    return masked_image

def normalize (image):
    normalized_image = np.zeros_like(image)
    min_val = np.min(image)
    max_val = np.max(image)
    scale = max_val - min_val
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                normalized_image[x, y, z] = int((image[x, y, z] - min_val) / (scale) + 0.6)
    return normalized_image

def adapthresh (image, step_size, t): # step_size = 3; t = 0.8
    int16_image = int16nize(image)
    normalized_image = normalize (int16_image)
    integral_image = calculate_integral_image(normalized_image)
    threshold_image = calculate_threshold_average(integral_image, step_size)
    print()
    print('finish adaptive thresholding!')
    masked_image = binarize (normalized_image, threshold_image, t)

    return masked_image

def calculate_mask_size (props):
    count_1 = 0
    count_100 = 0
    count_1000 = 0
    count_2000 = 0
    count_3000 = 0
    count_4000 = 0
    count_5000 = 0
    count_1_num = []
    count_100_num = []
    count_1000_num = []
    count_2000_num = []
    count_3000_num = []
    count_4000_num = []
    count_5000_num = []
    for i in range(len(props)):
        area = props[i].area
        if area > 0 and area < 100:
            count_1 += 1
            count_1_num.append(i)
        elif area >= 100 and area < 1000:
            count_100 += 1
            count_100_num.append(i)
        elif area >= 1000 and area < 2000:
            count_1000 += 1
            count_1000_num.append(i)
        elif area >= 2000 and area < 3000:
            count_2000 += 1
            count_2000_num.append(i)
        elif area >= 3000 and area < 4000:
            count_3000 += 1
            count_3000_num.append(i)
        elif area >= 4000 and area < 5000:
            count_4000 += 1
            count_4000_num.append(i)
        elif area >= 5000:
            count_5000 += 1
            count_5000_num.append(i)
    print()
    print('The mask sizes 👇')
    print('mask size between 1-100:', count_1)
    print(count_1_num)
    print('mask size between 100-1000:', count_100)
    print(count_100_num)
    print('mask size between 1000-2000:', count_1000)
    print(count_1000_num)
    print('mask size between 2000-3000:', count_2000)
    print(count_2000_num)
    print('mask size between 3000-4000:', count_3000)
    print(count_3000_num)
    print('mask size between 4000-5000:', count_4000)
    print(count_4000_num)
    print('mask size larger than 5000:', count_5000)
    print(count_5000_num)
    print()
    return count_1, count_100, count_1000, count_2000, count_3000, count_4000, count_5000

def calculate_gradient(image):
    gradient_magnitude = np.zeros_like(image, dtype=float)
    for axis in range(3):  # X, Y, and Z axes!!
        sobel_filtered = ndimage.sobel(image, axis=axis)
        gradient_magnitude += sobel_filtered**2
    gradient_magnitude = np.sqrt(gradient_magnitude)
    return gradient_magnitude




# FUNCTION FOR PRUNNING ===========================================================

def to_find_branch_and_noise (degree_0, degree_1, degree_2, degree_3, data):
    branch = []
    noise = []
    single_noise = []
    if len(degree_1) != 2:
        for coord in degree_1:
            partline = [coord]
            around_coordinates = count_occupied_neighbors (data, coord)
            new_coord = [element for element in around_coordinates if element not in partline][0]
            while new_coord in degree_2:
                partline.append(new_coord)
                new_around_coordinates = count_occupied_neighbors (data, new_coord)
                new_coord = [element for element in new_around_coordinates if element not in partline][0]
            if new_coord in degree_3:
                branch.append(partline)
            if new_coord in degree_1:
                partline.append(new_coord)
                noise.append(partline)
                degree_1.remove (new_coord)
    if len(degree_0) != 0:
        for coord in degree_0:
            single_noise.append(coord)
        noise.append(single_noise)
    if len(branch) == 0:
        print()
        print('Congrats! there is no branch anymore!')
    if len(noise) == 0:
        print()
        print('Congrats! there is no noise!')

    return branch, noise

def deleted_list_withvalue (branch, noise, branch_filtered_value):
    to_be_deleted = []
    # Firstly is to select the X% shortest branch in branches to delete

    sub_list_lengths = [len(sub_list) for sub_list in branch]
    # Sort sub-lists based on their lengths
    sorted_data = [sub_list for _, sub_list in sorted(zip(sub_list_lengths, branch))]
    # Determine the index for the shortest x_ws_ws_ws%
    index_threshold = int(len(sorted_data) * branch_filtered_value)
    # Select the shortest X% of sub-lists
    shortest_x_percent = sorted_data[:index_threshold]
    # Print the result
    for sub_list in shortest_x_percent:
        to_be_deleted +=sub_list

    # Second is to include all the noise
    for sub_list in noise:
        to_be_deleted +=sub_list

    return to_be_deleted

def preserved_points (data_to_be_deleted, original_data):

    # copy a resampled_skeleton_coordinates in order to maintain original values
    operated_data = original_data[:]

    # preserve
    final_data = [coord for coord in operated_data if coord not in data_to_be_deleted]

    return final_data

def plot_result (list, color, label, title):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = zip(*list)
    ax.scatter(x, y, z, c=color, marker='o', label=label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    ax.view_init(elev=20, azim=30)

    plt.show()

def count_degree(data, coord):
    x, y, z = coord
    degree = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                else:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    ncoordinate = np.array([nx, ny, nz])
                    is_coordinate_in_list = np.any(np.all(data == ncoordinate, axis=1))
                    if is_coordinate_in_list:
                        degree += 1
    return degree

def count_occupied_neighbors(data, coord):
    x, y, z = coord
    around_coordinates = [coord]
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                else:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    ncoordinate = np.array([nx, ny, nz])
                    is_coordinate_in_list = np.any(np.all(data == ncoordinate, axis=1))
                    if is_coordinate_in_list:
                        around_coordinates.append(ncoordinate)
                        around_coordinates = [item.tolist() if isinstance(item, np.ndarray) else item for item in around_coordinates]
    return around_coordinates

def is_list_contained(list1, list2):
    for coord in list1:
        if coord not in list2:
            return False
    return True

def triangle_pair_deleted (degree_2, degree_3, data):
    triangle_pair = []
    degree_3_copy = degree_3.copy()
    for coord in degree_3:
        neighbours = count_occupied_neighbors(data, coord)
        neighbour = neighbours.copy()
        neighbour.remove(coord)
        if is_list_contained(neighbour, degree_3):
            if coord in degree_3_copy:
                triangle_pair.append(coord)
                degree_3_copy = preserved_points (neighbours, degree_3_copy)

        else:
            for sub_coord in neighbour:
                if sub_coord in degree_2:
                    sub_neighbours = count_occupied_neighbors(data, sub_coord)
                    if is_list_contained(sub_neighbours, neighbours):
                        triangle_pair.append(sub_coord)

    return triangle_pair

def define_degree (list):
    degree_0 = []
    degree_1 = []
    degree_2 = []
    degree_3 = []
    for coord in list:
        degree = count_degree(list, coord)
        if degree == 0:
            degree_0.append(coord)
        elif degree == 1:
            degree_1.append(coord)
        elif degree == 2:
            degree_2.append(coord)
        elif degree >= 3:
            degree_3.append(coord)
    return degree_0, degree_1, degree_2, degree_3

def print_num(degree_0, degree_1, degree_2, degree_3, data):
    print('the coord num with degree = 0 is {}'.format(len(degree_0)))
    print('the coord num with degree = 1 is {}'.format(len(degree_1)))
    print('the coord num with degree = 2 is {}'.format(len(degree_2)))
    print('the coord num with degree = 3 is {}'.format(len(degree_3)))
    print('total coord preserved = {}'.format(len(data)))

def plot_seperated_result (degree_0, degree_1, degree_2, degree_3, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the coordinates from the first list
    if len(degree_0) != 0:
        x1, y1, z1 = zip(*degree_0)
        ax.scatter(x1, y1, z1, c='blue', marker='o', label='isoated points num={}'.format(len(degree_0)))

    # Plot the coordinates from the second list
    if len(degree_1) != 0:
        x2, y2, z2 = zip(*degree_1)
        ax.scatter(x2, y2, z2, c='cyan', marker='o', label='endpoints num={}'.format(len(degree_1)))

    # Plot the coordinates from the second list
    if len(degree_2) != 0:
        x3, y3, z3 = zip(*degree_2)
        ax.scatter(x3, y3, z3, c='green', marker='o', label='middle points num={}'.format(len(degree_2)))

    # Plot the coordinates from the second list
    if len(degree_3) != 0:
        x4, y4, z4 = zip(*degree_3)
        ax.scatter(x4, y4, z4, c='red', marker='o', label='branched points num={}'.format(len(degree_3)))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    # Customize the viewing angle if desired
    ax.view_init(elev=20, azim=30)

    plt.show()

def plot_seperated_result_triangle (degree_0, degree_1, degree_2, degree_3, triangle_list, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the coordinates from the first list
    if len(degree_0) != 0:
        x1, y1, z1 = zip(*degree_0)
        ax.scatter(x1, y1, z1, c='blue', marker='o', label='isoated points num={}'.format(len(degree_0)))

    # Plot the coordinates from the second list
    if len(degree_1) != 0:
        x2, y2, z2 = zip(*degree_1)
        ax.scatter(x2, y2, z2, c='cyan', marker='o', label='endpoints num={}'.format(len(degree_1)))

    # Plot the coordinates from the second list
    if len(degree_2) != 0:
        x3, y3, z3 = zip(*degree_2)
        ax.scatter(x3, y3, z3, c='green', marker='o', label='middle points num={}'.format(len(degree_2)))

    # Plot the coordinates from the second list
    if len(degree_3) != 0:
        x4, y4, z4 = zip(*degree_3)
        ax.scatter(x4, y4, z4, c='red', marker='o', label='branched points num={}'.format(len(degree_3)))

    # Plot the coordinates from the second list
    if len(triangle_list) != 0:
        x4, y4, z4 = zip(*triangle_list)
        ax.scatter(x4, y4, z4, c='orange', marker='o', label='triangle points num={}'.format(len(triangle_list)))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    # Customize the viewing angle if desired
    ax.view_init(elev=20, azim=30)

    plt.show()



# = FUNCTION FOR CHOOSING SPLINE POINTS===========================================================

def find_the_distance(points_list):
    distance = []
    length_of_spline = 0
    for i in range(len(points_list)-1):
        interval = np.sqrt((points_list[i][0]-points_list[i+1][0])**2 + (points_list[i][1]-points_list[i+1][1])**2 + (points_list[i][2]-points_list[i+1][2])**2)
        distance.append(interval)
        length_of_spline +=interval
    return distance

def arc_length(t, tck):

    def derivative(t, tck):
        der = splev(t, tck, der=1)
        return np.linalg.norm(der)

    integral, u = quad(derivative, 0, t, args=(tck,))
    return integral

def find_closest_point(ref_point, points):
    distances = [np.linalg.norm(np.array(ref_point) - np.array(coord)) for coord in points]
    closest_idx = np.argmin(distances)
    return points.pop(closest_idx)

# =Function of elongating the spline by Ditance_transform================================================

def dot_product(vector1, vector2):
    return sum(a*b for a, b in zip(vector1, vector2))

def magnitude(vector):
    return math.sqrt(sum(a**2 for a in vector))

def angle_between_vectors(vector1, vector2):
    if vector1[0] != 0 and vector1[1] != 0 and vector1[2] !=0 and vector2[0] != 0 and vector2[1] != 0 and vector2[2] !=0 and vector1[0] / vector2[0] == vector1[1] / vector2[1] == vector1[2] / vector2[2]:
        a = vector1[0] / vector2[0]
        if a < 0:
            angle = 180
        else:
            angle = 0
    else:
        dot_prod = dot_product(vector1, vector2)
        mag1 = magnitude(vector1)
        mag2 = magnitude(vector2)
        value = dot_prod / (mag1 * mag2)
        value = max(min(value, 1), -1)
        angle = math.degrees(math.acos(value))
    return angle

def to_find_next_coord (from_coord, to_coord, distance_transform):
    # define the radius
    B = np.array(from_coord)
    A = np.array(to_coord)
    direction_vector = B - A
    """ radius = np.linalg.norm(direction_vector) """
    radius = 10

    # define the bounding box
    x, y, z = [round(arr) for arr in to_coord]
    min_x = max(x - (1.5*radius), 0)
    max_x = min(x + (1.5*radius), distance_transform.shape[0]-1)
    min_y = max(y - (1.5*radius), 0)
    max_y = min(y + (1.5*radius), distance_transform.shape[1]-1)
    min_z = max(z - (1.5*radius), 0)
    max_z = min(z + (1.5*radius), distance_transform.shape[2]-1)

    coordinate_list = []

    # search the bounding box for all included
    for x_1 in range(int(min_x), int(max_x) + 1):
        for y_1 in range(int(min_y), int(max_y) + 1):
            for z_1 in range(int(min_z), int(max_z) + 1):
                if x_1 == x and y_1 == y and z_1 == z:
                    continue
                if distance_transform[x_1, y_1, z_1] != 0: # at least inside the mask
                        # print('distance_transform[x_1, y_1, z_1]',distance_transform[x_1, y_1, z_1])
                        vector = (x_1 - x, y_1 - y, z_1 -z)
                        distance_between = np.linalg.norm(vector)
                        # print(direction_vector, vector)
                        # print('distance',distance_between)
                        angle = angle_between_vectors(direction_vector, vector)
                        # print('angle=',angle)

                        if distance_between > (0.5 * 1) and distance_between < (3) and angle > 160:
                            coordinate_list.append([x_1, y_1, z_1])
    find_the_distance = 0
    find_the_coordinate = []
    # find the one with largest distance_transform
    if len(coordinate_list) != 0:
        for item in coordinate_list:
            X, Y, Z = item
            transformed_distance = distance_transform [X, Y, Z]
            if transformed_distance > find_the_distance:
                find_the_distance = transformed_distance
                find_the_coordinate = item
    # print('find_closest_point = ', find_the_coordinate)

    return find_the_coordinate

def find_the_coordinates_from_spline (new_points, num):
    coord = (new_points[0][num], new_points[1][num], new_points[2][num])
    return coord

def change_format (arr_coord):
    arr_coord = [float(arr) for arr in arr_coord]
    arr_coord = [round(arr,2) for arr in arr_coord]
    return arr_coord

def get_simple_list (list):

    simple_list = [tuple(item) if isinstance(item, np.ndarray) else item for item in list]
    simple_list = np.array (simple_list)
    return simple_list



# = FUNCTION FOR RESAMPLING===========================================================

# define a function to help with processing of skeleton coordinates list
def multiply_coordinates(coordinates, factors):
    multiplied_coordinates = [[x*factors[0], y*factors[1], z*factors[2]] for x, y, z in coordinates]
    return multiplied_coordinates

# define a function to help with processing of initial TIFF image
def image_preprocessing(path):

    # Read the TIFF file and convert it into a NumPy array
    with tifffile.TiffFile(path) as tif:
        # Get the first image in the TIFF file
        image = tif.asarray()
    # Now 'image' is a NumPy array containing the pixel data from the TIFF file
    image = np.array(image)
    transposed_image = np.transpose (image, axes=(1, 2, 0))

    return transposed_image


# = FUNCTION FOR Extracting intensity===========================================================

# Extract values at each coordinate, define a function to help with this process
def extract_pixels(image, points):
    values = []
    for i in range(points.shape[1]):
        x, y, z = points[:, i]
        # Assuming 'data' is your 3D numpy array with shape (111, 43, 29)
        value = image[int(x), int(y), int(z)]
        values.append(value)
    return values

# Extract values at each coordinate, define a function to help with this process
def extract_pixel(image, point):
    x, y, z = point
    # Assuming 'data' is your 3D numpy array with shape (111, 43, 29)
    value = image[int(x), int(y), int(z)]
    return value


# Functions for obtaining the masks

def obtain_mask(input_decov_path):
    image = image_preprocessing(input_decov_path)
    # Let's calculate the Global thresholding result
    global_image = global_threshold (image, 16000) #global_image

    # Let's calculate the noisy adaptive thresholding
    integral_image = calculate_integral_image(image)
    threshold_image = calculate_threshold_average(integral_image, 3) #variable
    binarize_image = binarize (image, threshold_image, 0.8, 1000) # variable

    # Let's calculate the gradient thresholding
    gradient_magnitude = calculate_gradient(image)
    flattened_data = gradient_magnitude.flatten()
    percentile_95_value = np.percentile(flattened_data, 95)  # this 95 is a variable
    gradient_image = global_threshold (gradient_magnitude, percentile_95_value)
    filled_mask = morphology.binary_fill_holes(gradient_image)

    # Let's use magic
    final_mask = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                adaptive_constrain = binarize_image[x, y, z] == 1 and filled_mask[x, y, z] ==1
                if global_image[x, y, z] == 1 or adaptive_constrain:
                    final_mask[x, y, z] = 1
                else:
                    final_mask[x, y, z] = 0

    """ scipy.io.savemat(save_path_mask, {'data': final_mask})
    mat_data = scipy.io.loadmat(save_path_mask)  """

    masked_image = final_mask.astype(int)
    from skimage.measure import label
    label_img = label(masked_image) # Lables the image based on their connectivity in 3D
    props = regionprops(label_img) # region measures

    return props


def screen_show_propsdetail(props):
    count = calculate_mask_size (props) # this would be shown on the screen
    return

def obtain_logical_images(k, props):
    # Obtain the mask data
    logical_mask = props[k].image.astype(int)
    return logical_mask

def obtain_fluorescent_image_stacks(path_for_intensity):
    image_stacks = []
    for i in range (len(path_for_intensity)):
        image_stack = image_preprocessing(path_for_intensity[i])
        image_stacks.append(image_stack)
    return image_stacks

def obtain_fluorescent_images(k, props, image_stacks):
    min_x, min_y, min_z, max_x, max_y, max_z = props[k].bbox
    intensity_images = []
    for i in range (len(image_stacks)):
        image_stack = image_stacks[i]
    # Prepare the cyan and magenta image input
        intensity_image = image_stack[min_x:max_x, min_y:max_y, min_z:max_z]
        intensity_images.append(intensity_image)
    return intensity_images

# preliminarily calculate branches and skeleton coordinates for filter
def calculate_branch(logical_mask):
    # Perform skeletonization on the 3D image
    skeleton = skeletonize_3d(logical_mask)
    skeleton_coordinates = np.array(np.where(skeleton)).T
    skeleton_coordinates = [list(array) for array in skeleton_coordinates]
    # determine degree
    degree_0, degree_1, degree_2, degree_3 = define_degree (skeleton_coordinates)
    if len(degree_0) != 0 or len(degree_1) != 2 or len(degree_3) != 0:
        branch_checkpoint = 1
    else:
        branch_checkpoint = 0
    return branch_checkpoint, skeleton_coordinates

# Calculate centoid
def calculate_centroid(k, props):
    centroid = props[k].centroid
    return centroid

# Calculate bbox
def calculate_bbox(k, props):
    bbox_size = props[k].bbox
    min_row, min_col, min_stack, max_row, max_col, max_stack = bbox_size
    return min_row, min_col, min_stack, max_row, max_col, max_stack

# Clculate the Vert and return it in string format
def calculate_verts(logical_mask):
    verts, faces, normals, values = measure.marching_cubes(logical_mask, level=0.5)
    return verts

# Calculate the skeleton thickness without elongation
def calculate_thickness(skeleton_coordinates, distance_transform):
    values = []
    for coord in skeleton_coordinates:
        thickness = distance_transform[coord[0], coord[1], coord[2]]
        values.append(thickness)
    thick_mean = statistics.mean(values)
    thick_variance = statistics.variance(values)
    return thick_mean

def calculate_thickness_real(skeleton_coordinates, distance_transform):
    shape = distance_transform.shape
    values = []
    for coord in skeleton_coordinates:
        thickness = distance_transform[min(round(coord[0]), shape[0]-1), min(round(coord[1]), shape[1]-1), min(round(coord[2]), shape[2]-1)]
        values.append(thickness*0.08)
    thickness = statistics.mean(values)
    thick_variance = statistics.variance(values)
    return thickness, thick_variance

# Englongate the ends and ontain the entire skeleton coordinates

def calculate_whole_skeleton(distance_transform, skeleton_coordinates):

    points_list_copy = copy.deepcopy(skeleton_coordinates)

    # set up reference point
    degree_0, degree_1, degree_2, degree_3 = define_degree (skeleton_coordinates)

    if len(degree_1) == 1:
        reference_point = degree_1[0]
    elif len(degree_1) == 0:
        reference_point = points_list_copy[0]
    else:
        reference_point = min(degree_1, key=lambda coord: coord[0])

    # Sorted list to store the ordered coordinates
    sorted_coordinates = []

    while points_list_copy:
        closest_point = find_closest_point(reference_point, points_list_copy)
        sorted_coordinates.append(closest_point)
        reference_point = closest_point

    final_coordinates = np.array (sorted_coordinates)

    # Calculate
    x, y, z = zip(*final_coordinates)
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    # Smoothing factor (s parameter)
    s = 20 # This can be changed!!!!
    points = np.array([x, y, z])
    # Perform spline interpolation
    tck, u = splprep(points, s=s)
    # Evaluate the spline on a new set of points
    new_points = splev(u, tck)

    coord_0 = change_format(np.array(new_points).T[0])
    coord_1 = change_format(np.array(new_points).T[1])
    coord_f1 = change_format(np.array(new_points).T[-1])
    coord_f2 = change_format(np.array(new_points).T[-2])

    # Begin elongating the ends
    skeleton_begin = []
    skeleton_end = []
    next_coord = to_find_next_coord (coord_1, coord_0, distance_transform)
    if next_coord != []:
        if distance_transform[next_coord[0], next_coord[1], next_coord[2]] > 0:
            skeleton_begin.append(next_coord)
            from_coord = coord_0
            to_coord = next_coord
            next_coord = to_find_next_coord (from_coord, to_coord, distance_transform)
            if next_coord != []:
                while distance_transform[next_coord[0], next_coord[1], next_coord[2]] > 0:
                    skeleton_begin.append(next_coord)
                    from_coord = to_coord
                    to_coord = next_coord
                    next_coord = to_find_next_coord (from_coord, to_coord, distance_transform)
                    if next_coord == []:
                        break

    next_coord = to_find_next_coord (coord_f2, coord_f1, distance_transform)
    if next_coord != []:
        if distance_transform[next_coord[0], next_coord[1], next_coord[2]] > 0:
            skeleton_end.append(next_coord)
            from_coord = coord_f1
            to_coord = next_coord
            next_coord = to_find_next_coord (from_coord, to_coord, distance_transform)
            if next_coord != []:
                while distance_transform[next_coord[0], next_coord[1], next_coord[2]] > 0:
                    skeleton_end.append(next_coord)
                    from_coord = to_coord
                    to_coord = next_coord
                    next_coord = to_find_next_coord (from_coord, to_coord, distance_transform)
                    if next_coord == []:
                        break

    # Get the whole skeleton list
    reversed_skeleton_begin = skeleton_begin[::-1]
    whole_skeleton = reversed_skeleton_begin + sorted_coordinates + skeleton_end

    # make sure there is no repetitive coordinate
    filtered_points = [whole_skeleton[0]]
    for point in whole_skeleton[1:]:
        if not np.array_equal(point, filtered_points[-1]):
            filtered_points.append(point)


    return filtered_points

# Calculate the total length
def calculate_length_and_curvatures(whole_skeleton):
    
    x, y, z = zip(*whole_skeleton)
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    # Smoothing factor (s parameter)
    s = 20
    # Prepare the data for splprep function
    points = np.array([x, y, z])
    # Perform spline interpolation
    tck, u = splprep(points, s=s)

    length_of_spline = arc_length(1, tck)


    # Initialize list to store evenly spaced points
    new_points = []

    # Calculate step size based on total arc length
    step_size = 0.5
    number_picked = round(length_of_spline/step_size)

    t=0
    # Generate evenly spaced points
    for i in range(number_picked):
        # print('we are now caltulating {}'.format(i+1))
        target_length = i * step_size
        while arc_length(t, tck) < target_length:
            t += 0.001  # Adjust the step size for accuracy
        point = splev(t, tck)
        # print(point)
        new_points.append(point)
    new_points = [[float(item) for item in sublist] for sublist in new_points]

    x_n, y_n, z_n = np.array(new_points).T
    # Prepare the data for splprep function
    points = np.array([x_n, y_n, z_n]) 

    tck, u = splprep(points, s=0)  # .T transposes the array to match splprep's expected input format

    # Evaluate the first and second derivatives
    first_derivatives = splev(u, tck, der=1)
    second_derivatives = splev(u, tck, der=2)

    # Compute the curvature at each point
    curvatures = []
    for i in range(len(u)):
        T = np.array(first_derivatives)[:, i]  # Tangent vector at this point
        T_prime = np.array(second_derivatives)[:, i]  # Derivative of tangent vector
        numerator = np.linalg.norm(np.cross(T, T_prime))
        denominator = np.linalg.norm(T) ** 3
        curvature = numerator / denominator
        curvatures.append(curvature)

    abs_curv = np.abs(curvatures) 
    max_curv = np.max(abs_curv)
    ave_curv = np.mean(abs_curv)
    std_curv = np.std(abs_curv)

    int_curv = 0
    # Loop through the list of points for calculating integral curvature
    for i in range(len(new_points) - 1):
        # Current point and the next point
        point_i = new_points[i]
        point_j = new_points[i + 1]
        
        # Calculate the distance between the points using the Euclidean distance formula
        distance = np.sqrt((point_j[0] - point_i[0])**2 + (point_j[1] - point_i[1])**2 + (point_j[2] - point_i[2])**2)
        int_curv += distance * abs_curv[i]

    normalized_int_curv = int_curv / length_of_spline



    return length_of_spline, curvatures, max_curv, ave_curv, std_curv, normalized_int_curv
        



def calculate_output_parameters(whole_skeleton, new_spacing, logical_mask, intensity_images): #new_spacing = (4, 4, 10)  # New voxel spacing along x, y, and z axes
    # Define the new target shape based on the desired real spacing
    original_spacing = (1, 1, 1)  # Original voxel spacing
    scale_factors = [new / original for new, original in zip(new_spacing, original_spacing)]

    # Resampling on the skeleton and logical mask ↓
    new_shape = [int(round(dim * scale_factor)) for dim, scale_factor in zip(logical_mask.shape, scale_factors)]

    # Perform the resampling using cubic spline interpolation
    resampled_mask = resize(logical_mask, new_shape, order=0)
    resampled_mask_coordinates = np.array(np.where(resampled_mask)).T
    resampled_skeleton = multiply_coordinates(whole_skeleton, scale_factors)

    # Separate the x, y, and z coordinates of the skeleton points
    x, y, z = zip(*resampled_skeleton)
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    # Smoothing factor (s parameter)
    s = 2000

    # Prepare the data for splprep function
    points = np.array([x, y, z])

    # Perform spline interpolation
    tck, u = splprep(points, s=s)
    length_of_spline = arc_length(1, tck)
    length = length_of_spline*0.08 # This is based on the real step
    # print
    print('calculated length')

    # Initialize list to store evenly spaced points
    new_points = []

    # Calculate step size based on total arc length
    step_size = 1.5

    number_picked = round(length_of_spline/step_size)

    t=0
    # Generate evenly spaced points
    for i in range(number_picked):
        # print('we are now caltulating {}'.format(i+1))
        target_length = i * step_size
        while arc_length(t, tck) < target_length:
            t += 0.001  # Adjust the step size for accuracy
        point = splev(t, tck)
        # print(point)
        new_points.append(point)
    new_points = [[float(item) for item in sublist] for sublist in new_points]


    x_n, y_n, z_n = np.array(new_points).T
    # Prepare the data for splprep function
    points = np.array([x_n, y_n, z_n]) 


    tck, u = splprep(points, s=0)  # .T transposes the array to match splprep's expected input format

    # Evaluate the first and second derivatives
    first_derivatives = splev(u, tck, der=1)
    second_derivatives = splev(u, tck, der=2)

    # Compute the curvature at each point
    curvatures = []
    for i in range(len(u)):
        T = np.array(first_derivatives)[:, i]  # Tangent vector at this point
        T_prime = np.array(second_derivatives)[:, i]  # Derivative of tangent vector
        numerator = np.linalg.norm(np.cross(T, T_prime))
        denominator = np.linalg.norm(T) ** 3
        curvature = numerator / denominator
        curvatures.append(curvature)

    abs_curv = np.abs(curvatures) 
    max_curv = np.max(abs_curv) 
    ave_curv = np.mean(abs_curv)
    std_curv = np.std(abs_curv)
    int_curv = 0
    # Loop through the list of points for calculating integral curvature
    for i in range(len(new_points) - 1):
        # Current point and the next point
        point_i = new_points[i]
        point_j = new_points[i + 1]
        
        # Calculate the distance between the points using the Euclidean distance formula
        distance = np.sqrt((point_j[0] - point_i[0])**2 + (point_j[1] - point_i[1])**2 + (point_j[2] - point_i[2])**2)
        int_curv += distance * abs_curv[i]

    normalized_int_curv = int_curv / length_of_spline

    # print
    print('calculated curvatures')

    # Calculate the thickness and thickness vaiance
    distance_transform = ndimage.distance_transform_edt(resampled_mask, sampling=(1, 1, 1), return_distances=True)
    thick_mean, thick_variance = calculate_thickness_real(new_points, distance_transform)

    x_h, y_h, z_h = np.transpose(new_points)

    # Combine x, y, and z into a single list
    new_points = [x_h, y_h, z_h]
    new_points = np.array(new_points).T

    # We are sorting the interval points ↓
    x_st, y_st, z_st = np.array(new_points[0])
    vector = new_points[0] - new_points[1]
    a1, b1, c1 = vector
    d1 = -(a1*x_st + b1*y_st + c1*z_st)
    A = [a1]
    B = [b1]
    C = [c1]
    D = [d1]

    for i in range (1, new_points.shape[0]):
        x, y, z = new_points[i]
        vector = new_points[i] - new_points[i-1]
        a, b, c = vector
        d = -(a*x + b*y + c*z)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)

        belonging_final = [[] for _ in range(new_points.shape[0]+1)]

    resampled_mask_coordinates = np.array(resampled_mask_coordinates)

    for i in range (resampled_mask_coordinates.shape[0]):
        x, y, z = resampled_mask_coordinates[i]
        belonging = []
        for j in range(new_points.shape[0]-1):
            a_pre = A[j]
            b_pre = B[j]
            c_pre = C[j]
            d_pre = D[j]
            a_post = A[j+1]
            b_post = B[j+1]
            c_post = C[j+1]
            d_post = D[j+1]
            if (a_pre*x+b_pre*y+c_pre*z+d_pre)*(a_post*x+b_post*y+c_post*z+d_post)<0 or a_pre*x+b_pre*y+c_pre*z+d_pre == 0:
                belonging.append(j)
        if (A[0]*x+B[0]*y+C[0]*z+D[0])*(A[1]*x+B[1]*y+C[1]*z+D[1])>0:
            belonging.append(0)
        if (A[new_points.shape[0]-2]*x+B[new_points.shape[0]-2]*y+C[new_points.shape[0]-2]*z+D[new_points.shape[0]-2])*(A[new_points.shape[0]-1]*x+B[new_points.shape[0]-1]*y+C[new_points.shape[0]-1]*z+D[new_points.shape[0]-1])>0:
            belonging.append(new_points.shape[0])
        if len(belonging)==1:
            num = belonging[0]
            belonging_final[num].append(resampled_mask_coordinates[i])
        elif len(belonging)>1:
            possible_belonging = []
            for k in belonging:
                if k == new_points.shape[0]:
                    possible_belonging.append(new_points[-1])
                else:
                    possible_belonging.append(new_points[k])
            possible_belonging = np.array(possible_belonging)
            distances = np.linalg.norm(possible_belonging - resampled_mask_coordinates[i], axis=1)
            min_distance_index = np.argmin(distances)
            num = belonging[min_distance_index]
            belonging_final[num].append(resampled_mask_coordinates[i])

    # print
    print('finish sorting points')

    # Resampling TIFF ↓
    resampled_images = []
    for image in intensity_images:
        # Calculate the new shape using the scale factors
        new_shape = [int(round(dim * scale_factor)) for dim, scale_factor in zip(image.shape, scale_factors)]
        # Perform the resampling using cubic spline interpolation
        resampled_image = resize(image, new_shape, order=3, mode='reflect', preserve_range=True)
        resampled_images.append(resampled_image)

    # Extracting intensity ↓
    final_intensities = []
    intensity_variances = []
    for image in resampled_images:
        # print
        print('start extracting image')
        final_values = []
        for interval in belonging_final:
            if len(interval) != 0:
                values = []
                for coordinate in interval:
                    value = extract_pixel(image, coordinate)
                    values.append(value)
                mean = np.mean(values)
                final_values.append(mean)
            else:
                mean = 0
                final_values.append(mean)
        final_values = fill_zeros(final_values)
        intensity_va = np.std(final_values)
        intensity_variances.append(intensity_va)
        final_intensities.append(final_values)


    return final_intensities, intensity_variances, length, max_curv, ave_curv, std_curv, normalized_int_curv, thick_mean, thick_variance


# Calculate the intensity from intensity_images
def fill_zeros(data):
    data = np.array(data)
    zero_indices = np.where(data == 0)[0]
    if len(zero_indices) != 0:
        for i in zero_indices:
            j = 1
            values = []
            while len(values)==0:
                i_pre = max(0, i-j)
                i_post = min(len(data)-1, i+j)
                values = []
                if data[i_pre] != 0:
                    values.append(data[i_pre])
                if data[i_post] != 0:
                    values.append(data[i_post])
                if len(values) != 0:
                    break
                if len(values) == 0:
                    j += 1
            data[i] = np.mean(values)
        data = data.tolist()
    return data

def calculate_intensity_and_variance(whole_skeleton, new_spacing, logical_mask, intensity_images): #new_spacing = (4, 4, 10)  # New voxel spacing along x, y, and z axes
    # Define the new target shape based on the desired real spacing
    original_spacing = (1, 1, 1)  # Original voxel spacing
    scale_factors = [new / original for new, original in zip(new_spacing, original_spacing)]

    # Resampling on the skeleton and logical mask ↓
    new_shape = [int(round(dim * scale_factor)) for dim, scale_factor in zip(logical_mask.shape, scale_factors)]

    # Perform the resampling using cubic spline interpolation
    resampled_mask = resize(logical_mask, new_shape, order=0)
    resampled_mask_coordinates = np.array(np.where(resampled_mask)).T
    resampled_skeleton = multiply_coordinates(whole_skeleton, scale_factors)

    # Separate the x, y, and z coordinates of the skeleton points
    x, y, z = zip(*resampled_skeleton)
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    # Smoothing factor (s parameter)
    s = 2000

    # Prepare the data for splprep function
    points = np.array([x, y, z])

    # Perform spline interpolation
    tck, u = splprep(points, s=s)
    length_of_spline = arc_length(1, tck)

    # Initialize list to store evenly spaced points
    new_points = []

    # Calculate step size based on total arc length
    step_size = 1.5

    number_picked = round(length_of_spline/step_size)

    t=0
    # Generate evenly spaced points
    for i in range(number_picked):
        # print('we are now caltulating {}'.format(i+1))
        target_length = i * step_size
        while arc_length(t, tck) < target_length:
            t += 0.001  # Adjust the step size for accuracy
        point = splev(t, tck)
        # print(point)
        new_points.append(point)
    new_points = [[float(item) for item in sublist] for sublist in new_points]

    x_h, y_h, z_h = np.transpose(new_points)

    # Combine x, y, and z into a single list
    new_points = [x_h, y_h, z_h]
    new_points = np.array(new_points).T

    # We are sorting the interval points ↓
    x_st, y_st, z_st = np.array(new_points[0])
    vector = new_points[0] - new_points[1]
    a1, b1, c1 = vector
    d1 = -(a1*x_st + b1*y_st + c1*z_st)
    A = [a1]
    B = [b1]
    C = [c1]
    D = [d1]

    for i in range (1, new_points.shape[0]):
        x, y, z = new_points[i]
        vector = new_points[i] - new_points[i-1]
        a, b, c = vector
        d = -(a*x + b*y + c*z)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)

        belonging_final = [[] for _ in range(new_points.shape[0]+1)]

    resampled_mask_coordinates = np.array(resampled_mask_coordinates)

    for i in range (resampled_mask_coordinates.shape[0]):
        x, y, z = resampled_mask_coordinates[i]
        belonging = []
        for j in range(new_points.shape[0]-1):
            a_pre = A[j]
            b_pre = B[j]
            c_pre = C[j]
            d_pre = D[j]
            a_post = A[j+1]
            b_post = B[j+1]
            c_post = C[j+1]
            d_post = D[j+1]
            if (a_pre*x+b_pre*y+c_pre*z+d_pre)*(a_post*x+b_post*y+c_post*z+d_post)<0 or a_pre*x+b_pre*y+c_pre*z+d_pre == 0:
                belonging.append(j)
        if (A[0]*x+B[0]*y+C[0]*z+D[0])*(A[1]*x+B[1]*y+C[1]*z+D[1])>0:
            belonging.append(0)
        if (A[new_points.shape[0]-2]*x+B[new_points.shape[0]-2]*y+C[new_points.shape[0]-2]*z+D[new_points.shape[0]-2])*(A[new_points.shape[0]-1]*x+B[new_points.shape[0]-1]*y+C[new_points.shape[0]-1]*z+D[new_points.shape[0]-1])>0:
            belonging.append(new_points.shape[0])
        if len(belonging)==1:
            num = belonging[0]
            belonging_final[num].append(resampled_mask_coordinates[i])
        elif len(belonging)>1:
            possible_belonging = []
            for k in belonging:
                if k == new_points.shape[0]:
                    possible_belonging.append(new_points[-1])
                else:
                    possible_belonging.append(new_points[k])
            possible_belonging = np.array(possible_belonging)
            distances = np.linalg.norm(possible_belonging - resampled_mask_coordinates[i], axis=1)
            min_distance_index = np.argmin(distances)
            num = belonging[min_distance_index]
            belonging_final[num].append(resampled_mask_coordinates[i])
    # Resampling TIFF ↓
    resampled_images = []
    for image in intensity_images:
        # Calculate the new shape using the scale factors
        new_shape = [int(round(dim * scale_factor)) for dim, scale_factor in zip(image.shape, scale_factors)]
        # Perform the resampling using cubic spline interpolation
        resampled_image = resize(image, new_shape, order=3, mode='reflect', preserve_range=True)
        resampled_images.append(resampled_image)

    # Extracting intensity ↓
    final_intensities = []
    intensity_variances = []
    for image in resampled_images:
        final_values = []
        for interval in belonging_final:
            if len(interval) != 0:
                values = []
                for coordinate in interval:
                    value = extract_pixel(image, coordinate)
                    values.append(value)
                mean = np.mean(values)
                final_values.append(mean)
            else:
                mean = 0
                final_values.append(mean)
        final_values = fill_zeros(final_values)
        intensity_va = np.std(final_values)
        intensity_variances.append(intensity_va)
        final_intensities.append(final_values)

    return final_intensities, intensity_variances

# Calculate the cell positions
from collections import Counter
def calculate_cell_position(skeleton_coordinates, min_row, min_col, min_stack, output_labeled_stack_path):
    # Read the TIFF file and convert it into a NumPy array
    with tifffile.TiffFile(output_labeled_stack_path) as tif:
        # Get the first image in the TIFF file
        stacked_image = tif.asarray()
    # Now 'image' is a NumPy array containing the pixel data from the TIFF file
    stacked_image = np.array(stacked_image)
    image = np.transpose (stacked_image, axes=(1, 2, 0))

    cell = []
    for point in skeleton_coordinates:
        x, y, z = point
        cell_num = image[int(x+min_row), int(y+min_col), int(z+min_stack)]
        cell.append(cell_num)

    # Filter out the zeros and count occurrences of each non-zero value
    non_zero_values = [value for value in cell if value != 0]
    if len(non_zero_values) == 0:
        formatted_output = [0, 0, 0, 0]
        portion_in_plasma = 0
    else:
        counts = Counter(non_zero_values)
        # Calculate total non-zero entries for percentage calculation
        total_non_zero = sum(counts.values())
        portion_in_plasma = (total_non_zero/len(cell))*100
        # Calculate the portion of each non-zero value in percentage
        portions = {value: (count / total_non_zero) * 100 for value, count in counts.items()}

        # Select the top 2 values and their portions
        top_2 = dict(Counter(portions).most_common(2))
        formatted_output = []
        for value, portion in top_2.items():
            formatted_output.append(value)
            formatted_output.append(int(portion))
        if len(formatted_output) == 2:
            formatted_output.append('0')
            formatted_output.append('0')
    # formatted_output = [top1, probability, top2, probability]
    # 'formatted_output' now contains the strings formatted as requested:
    return formatted_output, portion_in_plasma



# Show the final image

def show_image(final_intensities, final_ses, label_colors):

    #label_colors = ['c','m']
    single_figures = []
    for i in range (len(final_intensities)):
        # Create an array of indices to use as x-coordinates for the plot
        y = np.array(final_intensities[i])
        x= np.arange(len(y))
        se = np.array(final_ses[i])
        color = label_colors[i]

        # Plot the intensity, in two seperate image, or in one image
        # Interpolation
        x_smooth = np.linspace(x.min(), x.max(), (3*len(y)))
        y_smooth = make_interp_spline(x, y)(x_smooth)
        min_smooth = make_interp_spline(x, y-se)(x_smooth)
        max_smooth = make_interp_spline(x, y+se)(x_smooth)

        # Create a new figure and store it
        fig, ax = plt.subplots()
        ax.plot(x_smooth, y_smooth, color=color, label='Intensity')
        ax.fill_between(x_smooth, min_smooth, max_smooth, color=color, alpha=0.25, label='Standard Error')
        ax.legend()
        ax.set_title('Intensity Curve for Signal {}'.format(i+1))
        ax.set_xlabel('Length')
        ax.set_ylabel('Signal Intensity')
            # Store the figure object in the list
        single_figures.append(fig)

    merged_figure = []
    fig, ax1 = plt.subplots()

    y = final_intensities[0]
    x = np.arange(len(y))

    x_smooth = np.linspace(x.min(), x.max(), (3*len(y)))
    y_smooth = make_interp_spline(x, y)(x_smooth)

    ax1.set_xlabel('X-axis Label')
    ax1.set_ylabel('Cyan Fluorescent Intensity', color='black')
    ax1.plot(x_smooth, y_smooth, color=label_colors[0], label='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create the second plot (magenta)
    ax2 = ax1.twinx()
    y = final_intensities[1]
    x = np.arange(len(y))

    x_smooth = np.linspace(x.min(), x.max(), (3*len(y)))
    y_smooth = make_interp_spline(x, y)(x_smooth)
    ax2.set_ylabel('Magenta Fluorescent Intensity', color='black')
    ax2.plot(x_smooth, y_smooth, color=label_colors[1], label='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Show the plots
    plt.title('Merged Intensity')
    fig.tight_layout()
    plt.show()
    merged_figure.append(fig)

    return single_figures, merged_figure

def convert_coordinates_to_string(coords):
    return ', '.join("({:.2f}, {:.2f}, {:.2f})".format(*coord) for coord in coords)
def convert_values_to_string(values):
    # Convert each value to a string with desired formatting
    values_strings = ['{:.2f}'.format(value) for value in values]
    # Join all the string values into one large string
    values_str = ', '.join(values_strings)
    return values_str

def convert_values_to_string_more(values):
    # Convert each value to a string with desired formatting
    values_strings = ['{:.5f}'.format(value) for value in values]
    # Join all the string values into one large string
    values_str = ', '.join(values_strings)
    return values_str

def obtain_row(path_for_intensity, id, centroid, length_of_spline, thick_mean, thick_variance, min_row, min_col, min_stack, max_row, max_col, max_stack, whole_skeleton, duration, area, curvatures, max_curv, ave_curv, std_curv, normalized_int_curv, final_intensities, intensity_variances):

    skeleton_string = convert_coordinates_to_string(whole_skeleton)
    curvatures_string = convert_values_to_string_more(curvatures)
    # intensity2_string = convert_values_to_string(final_intensities[1])
    # vert_string = convert_coordinates_to_string(verts)
    row = [
        id,
        thick_mean,
        thick_variance,
        centroid[0],
        centroid[1],
        centroid[2],
        length_of_spline,
        min_row,
        min_col,
        min_stack,
        max_row,
        max_col,
        max_stack,
        skeleton_string,
        duration,
        area,
        curvatures_string, 
        max_curv, 
        ave_curv, 
        std_curv,
        normalized_int_curv
    ]

    for i in range(len(path_for_intensity)):
        intensity_string = convert_values_to_string(final_intensities[i])
        row.append(intensity_string)
        intensity_va = intensity_variances[i]
        row.append(intensity_va)

    return row


def function_one_signal (props, output_csv_path, path_for_intensity):
    image_stacks = obtain_fluorescent_image_stacks(path_for_intensity)
    for k in range(len(props)):
        try:
            # Start the timer
            start_time = time.time()

            if props[k].area > 100: # checkpoint1
                logical_mask = obtain_logical_images(k, props)
                branch_checkpoint, skeleton_coordinates = calculate_branch(logical_mask)
                if branch_checkpoint == 0 and logical_mask.shape[0]>2 and logical_mask.shape[1]>2 and logical_mask.shape[2]>2: # checkpoint2
                    if  len(skeleton_coordinates) >= 10: # checkpoint3

                        distance_transform = ndimage.distance_transform_edt(logical_mask, sampling=(1, 1, 1), return_distances=True)
                        thick_mean = calculate_thickness(skeleton_coordinates, distance_transform)
                        whole_skeleton = calculate_whole_skeleton(distance_transform, skeleton_coordinates)
                        length_of_spline, curvatures, max_curv, ave_curv, std_curv, normalized_int_curv = calculate_length_and_curvatures(whole_skeleton)

                        shape_constrain = (length_of_spline/thick_mean)
                        if shape_constrain >= 1: # checkpoint4
                            # A series of output ↓
                            print('pass all checkpoints 🥰')
                            centroid = calculate_centroid(k, props)
                            min_row, min_col, min_stack, max_row, max_col, max_stack = calculate_bbox(k, props)
                            intensity_images = obtain_fluorescent_images(k, props, image_stacks)
                            area = props[k].area
                            # Final extracting of the intensity 🥰
                            new_spacing = (4, 4, 10) # based on microscopy parameter of pixel width and z step
                            final_intensities, intensity_variances, length, max_curv, ave_curv, std_curv, normalized_int_curv, thick_mean, thick_variance = calculate_output_parameters(whole_skeleton, new_spacing, logical_mask, intensity_images)
                            # End the timer
                            end_time = time.time()
                            # Calculate the duration
                            duration = end_time - start_time

                            # Output
                            row = obtain_row(path_for_intensity, k, centroid, length, thick_mean, thick_variance, min_row, min_col, min_stack, max_row, max_col, max_stack, whole_skeleton, duration, area, curvatures, max_curv, ave_curv, std_curv, normalized_int_curv, final_intensities, intensity_variances)
                            with open(output_csv_path, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(row)
                        else:
                            print(k, 'short shape, not a XRI, remaining {}'.format(len(props)-k-1))
                    else:
                        print(k, 'short shape, not a XRI, remaining {}'.format(len(props)-k-1))
                else:
                    print(k, 'have branch, not a XRI, remaining {}'.format(len(props)-k-1))
            else:
                print(k, 'volume less than 100, not a XRI, remaining {}'.format(len(props)-k-1))

        except Exception as e:
            print("An error occurred at xri {}, error message: {}🫠".format(k, str(e)))
    print()
    print('Finish analyzing successfully 🙌')

        
## ================================================== Scripts ==========================================================

input_decov_path = '1-dec.tif' 
output_csv_path = '1.csv' 
green_channel= '1-green.tif' 
red_channel= '1-red.tif' 


print('we are analyzing')

path_for_intensity = []
path_for_intensity.append(green_channel)
path_for_intensity.append(red_channel)
header = ["ROI ID", 'Thickness_m', 'Thickness_v',"ROI_centroid_x", "ROI_centroid_y", "ROI_centroid_z", "Length",'ROI_minrow','ROI_mincol','ROI_minz', 'ROI_maxrow','ROI_maxcol','ROI_maxz', 'Skeleton_coords','Time', 'Volume', 'Curvatures', 'max_curvature','ave_curvature', 'std_curvature', 'normalized_integral_curvature']
header.append('Green_intensity')
header.append('Green_intensity_std')
header.append('Red_intensity')
header.append('Red_intensity_std')


props = obtain_mask(input_decov_path)
calculate_mask_size (props)

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

function_one_signal (props, output_csv_path, path_for_intensity)


