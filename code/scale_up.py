from PIL import Image
import os
import csv
# import String
import numpy as np
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import cv2


r_1024 = 6.0393258
r = r_1024/1024*1536
print(r) # 9.0589887


def crossProd(a, b, c):
    '''
    Outputs the crossproduct of two vectors formed by 3 coordinates.
    a, b, c: coordinates
    '''
    return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

start_1 = (256, 1280)
end_1 = (1280, 1280)
# point_1 = (256, 1280.005) # 5.12
point_1 = (256, 1280.9999) # 1023.8976 # use 1024
print(crossProd(start_1,end_1,point_1))


def circle1(x, y, sigma=9.0589887, step=2): # 6.0393258 + 0.5
    """
    Create a small collection of points in a neighborhood of some point.
    The number of points can be controlled by adjusting the step size.
    """
    # 1: 261; 2: 70; 3: 31
    # 2: 39; 3: 17; 4: 12
    neighborhood = []

    X = int(sigma)
    for i in range(-X, X + 1, step):  # Adjust step size to reduce the number of points
        Y = int(np.sqrt(sigma * sigma - i * i))
        for j in range(-Y, Y + 1, step):  # Adjust step size to reduce the number of points
            neighborhood.append((x + i, y + j))

    return neighborhood


def circle2(x, y, sigma=9.0589887): # 6.0393258 + 0.5
    """
    Create a small collection of points in a neighborhood of some point.
    The number of points can be controlled by adjusting the step size.
    """
    # 1: 261; 2: 70; 3: 31
    # 2: 39; 3: 17; 4: 12
    neighborhood = []

    X = int(sigma)
    for i in range(-X, X + 1):  # Adjust step size to reduce the number of points
        Y = int(np.sqrt(sigma * sigma - i * i))
        for j in range(-Y, Y + 1):  # Adjust step size to reduce the number of points
            neighborhood.append((x + i, y + j))

    return neighborhood

print(circle2(0,0))
print(len(circle2(0,0))) # 261


def visualize_circle_points(x, y, sigma):
    """
    Generate and visualize points using the circle function.

    Parameters:
    - x, y: Center coordinates of the circle.
    - sigma: Defines the radius of the circle.
    """
    # Generate points using the circle function
    points = circle2(x, y, sigma)

    # Extract x and y coordinates for plotting
    x_coords, y_coords = zip(*points)

    # Plotting the points
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Points Generated by Circle Function with sigma={sigma}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

# Example usage
visualize_circle_points(0, 0, sigma=9.0589887)


assert(r==2)
#
# def stretch_image(input_path, output_path):
#     # Open the image
#     with Image.open(input_path) as img:
#         # Get original dimensions
#         width, height = img.size
#
#         # Set new dimensions (stretch width by 2x)
#         new_width = width * 2
#         new_size = (new_width, height)
#
#         # Resize the image
#         stretched_img = img.resize(new_size, Image.BICUBIC)
#
#         # Save the stretched image
#         stretched_img.save(output_path)
#
#
# def check_size_768(image_dir):  # return True if size = (768, 768)
#     with Image.open(image_dir) as img:
#         # Get the image size
#         width, height = img.size
#         if (width, height) != (768, 768):
#             return False
#         return True
#
# visit_1_dir = 'Visit_1_extracted'
# visit_2_dir = 'Visit_2_extracted'
#
# # List of the two directories to loop through
# directories = [visit_1_dir, visit_2_dir]
#
#
# # --------------------------------------------------
# # 1. scaling up B-scan images from 512*496 to 1024*496
#
# # Loop over each directory
# for parent_dir in directories:
#     # Loop over each folder within the current parent directory
#     for E2E_folder in os.listdir(parent_dir):
#         E2E_path = os.path.join(parent_dir, E2E_folder)
#         if not os.path.isdir(E2E_path):  # Skip if it's not a directory
#             continue
#         # Skip if there's a folder stretched_volume_0 inside E2E_path
#         if os.path.isdir(os.path.join(E2E_path, 'stretched_volume_0')):
#             continue
#
#         oct_dir = os.path.join(E2E_path, "images", "2D_0.png")
#         if not check_size_768(oct_dir):  # Skip if oct_dir is not 768*768
#             continue
#
#         # Create a folder stretched_volume_0 inside E2E_path
#         # Construct the full path for the new folder
#         new_folder_path = os.path.join(E2E_path, 'stretched_volume_0')
#         # Create the new folder
#         os.makedirs(new_folder_path, exist_ok=True)
#         print(f"Folder created at: {new_folder_path}")
#
#         old_volume_path = os.path.join(E2E_path, "volume_0")
#         for filename in os.listdir(old_volume_path):
#             old_image_path = os.path.join(old_volume_path, filename)
#             new_image_path = os.path.join(new_folder_path, filename)
#             stretch_image(old_image_path, new_image_path)
#
# # Folder created at: Visit_1_extracted/029_OS.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_1_extracted/055_OD_49.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_1_extracted/055_OS_49.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_1_extracted/029_OD.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_2_extracted/042_OS.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_2_extracted/024_OD.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_2_extracted/023_OS.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_2_extracted/024_OS.E2E_extracted/stretched_volume_0
# # Folder created at: Visit_2_extracted/023_OD.E2E_extracted/stretched_volume_0
#
#
#
#
# # --------------------------------------------------
# # 2. scaling up slo coordinates csv
#
#
# # slo_coord_csv_filename = "SLO_coordinates_0.csv"
# # new_slo_coord_csv_filename = "stretched_SLO_coordinates_0.csv"
# # new_slo_csv_path = os.path.join(E2E_path, new_slo_coord_csv_filename)
#
# # Loop over each directory
# for parent_dir in directories:
#     # Loop over each folder within the current parent directory
#     for E2E_folder in os.listdir(parent_dir):
#         E2E_path = os.path.join(parent_dir, E2E_folder)
#         if not os.path.isdir(E2E_path):  # Skip if it's not a directory
#             continue
#         # Skip if there isn't a folder stretched_volume_0 inside E2E_path
#         if not os.path.isdir(os.path.join(E2E_path, 'stretched_volume_0')):
#             continue
#
#         slo_coord_csv_filename = "SLO_coordinates_0.csv"
#         new_slo_coord_csv_filename = "stretched_SLO_coordinates_0.csv"
#         old_slo_csv_path = os.path.join(E2E_path, slo_coord_csv_filename)
#         new_slo_csv_path = os.path.join(E2E_path, new_slo_coord_csv_filename)
#
#         # Skip if there is a csv file called stretched_SLO_coordinates_0.csv
#         if os.path.exists(new_slo_csv_path):
#             continue
#
#         # Open the input CSV file for reading
#         with open(old_slo_csv_path, mode='r', newline='') as infile:
#             reader = csv.reader(infile)
#             # Read the header
#             header = next(reader)
#
#             # Open the output CSV file for writing
#             with open(new_slo_csv_path, mode='w', newline='') as outfile:
#                 writer = csv.writer(outfile)
#
#                 # Write the header to the output CSV
#                 writer.writerow(header)
#
#                 # Iterate over each row in the input CSV
#                 for row in reader:
#                     # Convert each value in the row to an integer, double it, and then convert it back to string
#                     doubled_row = [str(int(value) * 2) for value in row]
#                     # Write the doubled values to the output CSV
#                     writer.writerow(doubled_row)
#
#         print(f"Doubled values have been written to {new_slo_csv_path}")
#
#
#
#
# # --------------------------------------------------
# # 3. scaling up mp coords csvs in Registered_coord folder
#
# registered_dir = 'Registered'
# registered_coord_dir = 'Registered_coords'
# # print('registered_76_1062'[10:])  # _76_1062
#
# for image in os.listdir(registered_dir):
#     image_path = os.path.join(registered_dir, image)
#     if image[0] != "r":  # Skip if the image is not registered
#         continue
#     if not check_size_768(image_path):  # Skip if the image's size is not 768*768
#         continue
#     # Record name[10:] ?
#     tail_numbers = image[10:-4]  # _76_1062
#     old_mp_coord_csv_name = 'MP'+tail_numbers+'.csv'
#     new_mp_coord_csv_name = 'stretched_MP'+tail_numbers+'.csv'
#
#     old_mp_csv_path = os.path.join(registered_coord_dir, old_mp_coord_csv_name)
#     new_mp_csv_path = os.path.join(registered_coord_dir, new_mp_coord_csv_name)
#
#     # Skip if there's a stretched csv for i
#     if os.path.exists(new_mp_csv_path) and os.path.isfile(new_mp_csv_path):
#         continue
#
#     # Open the old CSV file for reading
#     with open(old_mp_csv_path, mode='r', newline='') as infile:
#         reader = csv.reader(infile)
#
#         # Open the new CSV file for writing
#         with open(new_mp_csv_path, mode='w', newline='') as outfile:
#             writer = csv.writer(outfile)
#
#             # Iterate over each row in the old CSV
#             for row in reader:
#                 # Convert each value in the row to a float, double it, and then format it as a string
#                 doubled_row = [f"{float(value) * 2:.17e}" for value in row]
#                 # Write the doubled values to the new CSV
#                 writer.writerow(doubled_row)
#
#     print(f"Doubled values have been written to {new_mp_csv_path}")
#
# # Doubled values have been written to Registered_coords/stretched_MP_153_2218.csv
# # Doubled values have been written to Registered_coords/stretched_MP_106_651.csv
# # Doubled values have been written to Registered_coords/stretched_MP_65_1403.csv
# # Doubled values have been written to Registered_coords/stretched_MP_106_650.csv
# # Doubled values have been written to Registered_coords/stretched_MP_65_1404.csv
# # Doubled values have been written to Registered_coords/stretched_MP_76_1602.csv
# # Doubled values have been written to Registered_coords/stretched_MP_76_1603.csv
# # Doubled values have been written to Registered_coords/stretched_MP_195_2013.csv
# # Doubled values have been written to Registered_coords/stretched_MP_195_2012.csv




# --------------------------------------------------
# copies of functions here:
def crossProd(a, b, c):
    '''
    Outputs the crossproduct of two vectors formed by 3 coordinates.
    a, b, c: coordinates
    '''
    return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

def formatCoords(row):
    return ([row[0], row[2]], [row[1], row[3]])

def getBscan(img_num):
    '''
    Get the name of the B-scan that intersects the MP point
    '''
    if img_num < 10:
        bscan_name = "volume_00"+str(img_num)+".png"
    elif 10 <= img_num < 100:
        bscan_name = "volume_0"+str(img_num)+".png"
    else:
        bscan_name = "volume_"+str(img_num)+".png"
    return bscan_name

def getHypotheneuse(coordinate, start_coordinate, end_coordinate): # for OCT B-scans that are slanted
    width = abs(coordinate[0]-start_coordinate[0])
    height = abs(end_coordinate[1]-start_coordinate[1])
    return int(math.sqrt(width**2 + height**2))

def getSeg(coordinate, image):
    '''
    Get slice of OCT image
    '''
    c_min = coordinate-16
    c_max =coordinate+16
    segment = image[:,c_min:c_max]
    return segment

def circle(x, y, sigma=13):
    """ create a small collection of points in a neighborhood of some point
    """
    neighborhood = []

    X = int(sigma)
    for i in range(-X, X + 1):
        Y = int(pow(sigma * sigma - i * i, 1/2))
        for j in range(-Y, Y + 1):
            neighborhood.append((x + i, y + j))

    return neighborhood

def noOverlap(x, y, sigma=13):
    return [(x,y),(x,y+sigma/4),(x,y-sigma/4),(x,y+sigma/2),(x,y-sigma/2),(x,y-sigma/4*3),(x,y-sigma/4*3),(x,y-sigma),(x,y-sigma)]


def proportional_distance(a, b, c):
    """
    Calculates the proportional distance from point c to the line connecting a and b.
    """
    # Calculate the length of the line segment
    line_length = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    # Calculate the perpendicular distance, normalized by line length
    distance = abs(c[1] - a[1]) / line_length

    return distance

# Example usage with the three sets of coordinates
start_1 = (256, 1280)
end_1 = (1280, 1280)
point_1 = (256, 1280.9)  # 0.000879
# # Example usage with the three sets of coordinates
# start_1 = (256, 1280)
# end_1 = (1280, 1280)
# point_1 = (256, 1280.586)
#
# start_2 = (128, 640)
# end_2 = (640, 640)
# point_2 = (128, 640.293)
#
# start_3 = (120, 644)
# end_3 = (632, 644)
# point_3 = (130, 644.293)
#
# # Calculate proportional distances for all three cases
distance_1 = proportional_distance(start_1, end_1, point_1)
# distance_2 = proportional_distance(start_2, end_2, point_2)
# distance_3 = proportional_distance(start_3, end_3, point_3)
#
print("distance_1, distance_2, distance_3: ")
print(f"{distance_1:.10e}")
# print(f"{distance_2:.10e}")
# print(f"{distance_3:.10e}")
# # 0.000572


# # --------------------------------------------------
# # Test: check if intersections tested before == intersections tested now
#
# test_dir = "test_folder/029_OS.E2E_extracted"  # 029 OS v1 - 106_651
#
# MPinfo = pd.read_csv(test_dir+'/MP_centres.csv')
# MPvalues = genfromtxt(test_dir+'/MP_values.csv', dtype=int, delimiter=',', encoding='utf-8-sig')
#
# test_volume0folder_dir = os.path.join(test_dir, "volume_0")
# test_stretchedvolume0folder_dir = os.path.join(test_dir, "stretched_volume_0")
#
# test_mpcoord_dir = os.path.join(test_dir, "MP_106_651.csv")
# test_stretchedmpcoord_dir = os.path.join(test_dir, "stretched_MP_106_651.csv")
# test_SLOcoord_dir = os.path.join(test_dir, "SLO_coordinates_0.csv")
# test_stretchedSLOcoord_dir = os.path.join(test_dir, "stretched_SLO_coordinates_0.csv")
#
# mp_values = MPvalues[:,12] # column 12 is 029 OS v1
#
# # print coordinates of mp points that are at intersections
#
# # before:
# intersection_coords1 = []
# intersection_mpvalue1 = []
# intersection_bscan1 = []
# oct_info1 = genfromtxt(test_SLOcoord_dir, delimiter=',', dtype=int, encoding='utf-8-sig')[1:,]
# mp_points1 = genfromtxt(test_mpcoord_dir, dtype=float, delimiter=',', encoding='utf-8-sig')
# oct_coords1 = np.apply_along_axis(formatCoords, 1, oct_info1)
# assert(len(mp_points1) == len(mp_values))
#
# # # circle
# # for j, mp_centre in enumerate(mp_points1):  # each point
# #     if mp_values[j] != -1:
# #         oversample = circle(mp_centre[0], mp_centre[1], sigma=6.5)
# #         for m, mp_coords in enumerate(oversample):
# #             #print(mp_coords)
# #             for i, oct_coord in enumerate(oct_coords1):
# #                 #print(oct_coord)
# #                 start_coords = oct_coord[0]
# #                 end_coords = oct_coord[1]
# #                 assert(start_coords[0] != end_coords[0])
# #                 # if abs(crossProd(start_coords, end_coords, mp_coords)) < 600: # margin of <1 pixels away for human error/rounding
# #                 if abs(proportional_distance(start_coords, end_coords, mp_coords)) < 0.000572:
# #                     intersection_coords1.append(mp_coords)
# #                     intersection_mpvalue1.append(mp_values[j])
# #                     bscan_name = getBscan(i)
# #                     intersection_bscan1.append(bscan_name)
#
# for j, mp_centre in enumerate(mp_points1):  # each point
#     if mp_values[j] != -1:
#         # oversample = circle(mp_centre[0], mp_centre[1], sigma=6.5)
#         for i, oct_coord in enumerate(oct_coords1):
#             #print(oct_coord)
#             start_coords = oct_coord[0]
#             end_coords = oct_coord[1]
#             assert(start_coords[0] != end_coords[0])
#             mp_coords = mp_centre[0], mp_centre[1]
#             if abs(proportional_distance(start_coords, end_coords, mp_coords)) < 0.5:
#                 intersection_coords1.append(mp_coords)
#                 intersection_mpvalue1.append(mp_values[j])
#                 bscan_name = getBscan(i)
#                 intersection_bscan1.append(bscan_name)
#
# print("intersection_coords1: ")
# print(intersection_coords1)
# print("intersection_mpvalue1: ")
# print(intersection_mpvalue1)
# print("intersection_bscan1: ")
# print(intersection_bscan1)
#
#
#
# # now: (stretched)
# intersection_coords2 = []
# intersection_mpvalue2 = []
# intersection_bscan2 = []
# oct_info2 = genfromtxt(test_stretchedSLOcoord_dir, delimiter=',', dtype=int, encoding='utf-8-sig')[1:,]
# mp_points2 = genfromtxt(test_stretchedmpcoord_dir, dtype=float, delimiter=',', encoding='utf-8-sig')
# oct_coords2 = np.apply_along_axis(formatCoords, 1, oct_info2)
# assert(len(mp_points2) == len(mp_values))
#
# # for j, mp_centre in enumerate(mp_points2):  # each point
# #     if mp_values[j] != -1:
# #         oversample = circle(mp_centre[0], mp_centre[1])
# #         for m, mp_coords in enumerate(oversample):
# #             #print(mp_coords)
# #             for i, oct_coord in enumerate(oct_coords2):
# #                 #print(oct_coord)
# #                 start_coords = oct_coord[0]
# #                 end_coords = oct_coord[1]
# #                 assert(start_coords[0] != end_coords[0])
# #                 # if abs(crossProd(start_coords, end_coords, mp_coords)) < 600: # margin of <1 pixels away for human error/rounding
# #                 if abs(proportional_distance(start_coords, end_coords, mp_coords)) < 0.000572:
# #                     intersection_coords2.append(mp_coords)
# #                     intersection_mpvalue2.append(mp_values[j])
# #                     bscan_name = getBscan(i)
# #                     intersection_bscan2.append(bscan_name)
#
# for j, mp_centre in enumerate(mp_points2):  # each point
#     if mp_values[j] != -1:
#         # oversample = circle(mp_centre[0], mp_centre[1], sigma=6.5)
#         for i, oct_coord in enumerate(oct_coords2):
#             #print(oct_coord)
#             start_coords = oct_coord[0]
#             end_coords = oct_coord[1]
#             assert(start_coords[0] != end_coords[0])
#             mp_coords = mp_centre[0], mp_centre[1]
#             if abs(proportional_distance(start_coords, end_coords, mp_coords)) < 0.5:  # 0.000572
#                 intersection_coords2.append(mp_coords)
#                 intersection_mpvalue2.append(mp_values[j])
#                 bscan_name = getBscan(i)
#                 intersection_bscan2.append(bscan_name)
#
# print("intersection_coords2: ")
# print(intersection_coords2)
# print("intersection_mpvalue2: ")
# print(intersection_mpvalue2)
# print("intersection_bscan2: ")
# print(intersection_bscan2)
#
# print(len(intersection_coords1) == len(intersection_coords2))
# print(len(intersection_mpvalue1) == len(intersection_mpvalue2))
# print(len(intersection_bscan1) == len(intersection_bscan2))
#
#
# print(len(intersection_coords1))  # 747
# print(len(intersection_coords2))  # 582
# print(len(intersection_mpvalue1))
# print(len(intersection_mpvalue2))
# print(len(intersection_bscan1))
# print(len(intersection_bscan2))


# distance
start_coords=(256,1280)
end_coords=(1280,1280)
mp_coords=(256,1280.586)
print(abs(crossProd(start_coords, end_coords, mp_coords)))  # 600
print(abs(proportional_distance(start_coords, end_coords, mp_coords)))  # 0.0005722656250000124
##!

start_coords=(256,1280)
end_coords=(1280,1280)
mp_coords=(256,1281)
print(abs(crossProd(start_coords, end_coords, mp_coords)))  # 1024
print(abs(proportional_distance(start_coords, end_coords, mp_coords)))  # 0.0009765625


start_coords=(256,1280)
end_coords=(1280,1280)
mp_coords=(256,1282.441)
print(abs(crossProd(start_coords, end_coords, mp_coords)))  # 2500
print(abs(proportional_distance(start_coords, end_coords, mp_coords)))  # 0.00238378906250003
##!




# count the total number of volumes

visit_1_dir = 'Visit_1_extracted'
visit_2_dir = 'Visit_2_extracted'

# List of the two directories to loop through
directories = [visit_1_dir, visit_2_dir]
#
# # Loop over each directory
# count_volumes = 0
# for parent_dir in directories:
#     # Loop over each folder within the current parent directory
#     for E2E_folder in os.listdir(parent_dir):
#         E2E_path = os.path.join(parent_dir, E2E_folder)
#         if not os.path.isdir(E2E_path):  # Skip if it's not a directory
#             continue
#         volume0folder_dir = E2E_path + "/volume_0"
#         for volumes in os.listdir(volume0folder_dir):
#             count_volumes += 1
#
# print(count_volumes)  # 4308


# Record paths for all the volumes
list_of_volumes_path = []
for parent_dir in directories:
    # Loop over each folder within the current parent directory
    for E2E_folder in os.listdir(parent_dir):
        E2E_path = os.path.join(parent_dir, E2E_folder)
        if not os.path.isdir(E2E_path):  # Skip if it's not a directory
            continue
        volume0folder_dir = E2E_path + "/volume_0"
        stretchedvolume0folder_dir = E2E_path + "/stretched_volume_0"
        # if stretched exists, use volumes in stretched_volume_0 folder
        if os.path.exists(stretchedvolume0folder_dir) and os.path.isdir(stretchedvolume0folder_dir):
            which_volume0folder_dir = stretchedvolume0folder_dir
        else:
            which_volume0folder_dir = volume0folder_dir
        for volumes in os.listdir(which_volume0folder_dir):
            volume_path = which_volume0folder_dir + "/" + volumes
            list_of_volumes_path.append(volume_path)

# print(len(list_of_volumes_path))  # 4308

# print(list_of_volumes_path[0])
# Visit_1_extracted/042_OD_49.E2E_extracted/volume_0/volume_039.png
# 'Visit_2_extracted/023_OD.E2E_extracted/stretched_volume_0/volume_014.png',...

# Randomly assign volumes into train/val/test lists with 80/10/10

train_vol_list = []
val_vol_list = []
test_vol_list = []

# assign
for eachvolume in list_of_volumes_path:
    class_point = np.random.rand(1)
    if class_point < 0.8:  # train
        train_vol_list.append(eachvolume)
    elif class_point >= 0.8 and class_point < 0.9:  # val
        val_vol_list.append(eachvolume)
    else:  # test
        test_vol_list.append(eachvolume)

print(len(train_vol_list))  # 3473
print(len(val_vol_list))  # 402
print(len(test_vol_list))  # 433

# ratio of train/all, validation/all, test/all
print("ratio of train/all, validation/all, test/all")
print(float(len(train_vol_list)/4308))
print(float(len(val_vol_list)/4308))
print(float(len(test_vol_list)/4308))
# 0.8061745589600743
# 0.09331476323119778
# 0.10051067780872795


# radius of an MP point:
diameter_in_um = 129
radius_in_um = diameter_in_um/2
radius_in_px = radius_in_um/10.68
R = radius_in_px # 6.0393258

def create_circle(x, y, sigma=R+0.5):
    """ create a small collection of points in a neighborhood of some point
    """
    neighborhood = []

    X = int(sigma)
    for i in range(-X, X + 1):
        Y = int(pow(sigma * sigma - i * i, 1/2))
        for j in range(-Y, Y + 1):
            neighborhood.append((x + i, y + j))

    return neighborhood

### edit below
#
bscan_name = getBscan(i)
which_volume_folder = '/volume_0/'  # oct_dir = E2E_path here
if os.path.exists(oct_dir + '/stretched_volume_0') and os.path.isdir(oct_dir + '/stretched_volume_0'):
    which_volume_folder = '/stretched_volume_0/'
bscan_path = oct_dir + which_volume_folder + bscan_name
# TODO: change oct_dir + which_volume_folder + bscan_name

def save_directory(bscan_path, train_vol_list=train_vol_list,
                   val_vol_list=val_vol_list, test_vol_list=test_vol_list):
    # Splitting data into train/val/test
    if bscan_path in train_vol_list:
        save_dir = 'Train/train'
    elif bscan_path in val_vol_list:
        save_dir = 'Validate/val'
    elif bscan_path in test_vol_list:
        save_dir = 'Test/test'
    else:
        print(str(bscan_path)+" is not in any lists")
        return None
    return save_dir

bscan = cv2.imread(bscan_path)  # read .png

######
# skip if mp value is -1
# if mp_values[j] == -1:
#     continue

# edit
save_dir = save_directory(bscan_path=bscan_path)
if save_dir == 'Train/train':
    pass
elif save_dir == 'Validate/val':
    pass
else:
    assert save_dir == 'Test/test'
    pass


# check the length of train, val, test - slices
print("amount of slices in train, val and test sets:")
print(train_num)
print(val_num)
print(test_num)

total_slices = train_num + val_num + test_num
print("ratio of slices in train/all, validation/all, test/all")
print(float(train_num/total_slices))
print(float(val_num/total_slices))
print(float(test_num/total_slices))

