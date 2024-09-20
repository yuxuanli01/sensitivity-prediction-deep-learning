import numpy as np
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import csv
import os
import time #


# Record the start time
start_time = time.time()


### Import data ###
MPinfo1 = pd.read_csv('MP_centres.csv')
MPvalues1 = genfromtxt('MP_values.csv', dtype=int, delimiter=',', encoding='utf-8-sig')

# removed 019 OS 1
MPinfo = MPinfo1.drop(index=0).reset_index(drop=True)
MPvalues = MPvalues1[:, 1:]
oct_dir_list = MPinfo['OCT_folder'].tolist()

# ->
visit_1_dir = 'Visit_1_extracted'
visit_2_dir = 'Visit_2_extracted'
directories = [visit_1_dir, visit_2_dir]

skip_E2E_list = ['019_OD.E2E_extracted', '019_OS.E2E_extracted']
skip_volume_list = ['Visit_2_extracted/049_OS.E2E_extracted/volume_0/volume_006.png']
# skip_volume_list = ['Visit_2_extracted/049_OS.E2E_extracted/normalized_volume_0/volume_006.png']

# Record paths for all the volumes
list_of_volumes_path = []
for parent_dir in directories:
    # Loop over each folder within the current parent directory
    for E2E_folder in os.listdir(parent_dir):
        if E2E_folder in skip_E2E_list:
            continue
        E2E_path = os.path.join(parent_dir, E2E_folder)
        if not os.path.isdir(E2E_path):  # Skip if it's not a directory
            continue
        # Skip if path is not in MP_centres
        if E2E_path not in oct_dir_list:
            continue
        volume0folder_dir = E2E_path + "/volume_0"
        stretchedvolume0folder_dir = E2E_path + "/stretched_volume_0"
        # # if stretched exists, use volumes in stretched_volume_0 folder
        # if os.path.exists(stretchedvolume0folder_dir) and os.path.isdir(stretchedvolume0folder_dir):
        #     which_volume0folder_dir = stretchedvolume0folder_dir
        # else:
        #     which_volume0folder_dir = volume0folder_dir
        # for volumes in os.listdir(which_volume0folder_dir):
        #     volume_path = which_volume0folder_dir + "/" + volumes
        #     list_of_volumes_path.append(volume_path)

        # volume0folder_dir = E2E_path + "/normalized_volume_0"
        # if stretched exists, use volumes in stretched_volume_0 folder
        if os.path.exists(stretchedvolume0folder_dir) and os.path.isdir(stretchedvolume0folder_dir):
            which_volume0folder_dir = stretchedvolume0folder_dir
        else:
            which_volume0folder_dir = volume0folder_dir
        for volumes in os.listdir(which_volume0folder_dir):
            volume_path = which_volume0folder_dir + "/" + volumes
            list_of_volumes_path.append(volume_path)
for item in skip_volume_list:
    list_of_volumes_path.remove(item)
print(len(list_of_volumes_path))  # 4014. before: 4308

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
# print(len(train_vol_list))  # 3473
# print(len(val_vol_list))  # 402
# print(len(test_vol_list))  # 433
# ratio of train/all, validation/all, test/all
print("ratio of volumes in train/all, validation/all, test/all")
print(len(train_vol_list), float(len(train_vol_list)/len(list_of_volumes_path)))
print(len(val_vol_list), float(len(val_vol_list)/len(list_of_volumes_path)))
print(len(test_vol_list), float(len(test_vol_list)/len(list_of_volumes_path)))
# 0.8061745589600743
# 0.09331476323119778
# 0.10051067780872795

# <-

def crossProd(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

def formatCoords(row):
    return ([row[0], row[2]], [row[1], row[3]])

def getBscan(img_num):
    if img_num < 10:
        bscan_name = "volume_00"+str(img_num)+".png"
    elif 10 <= img_num < 100:
        bscan_name = "volume_0"+str(img_num)+".png"
    elif img_num >= 100:
        bscan_name = "volume_"+str(img_num)+".png"
    else:
        print(str(img_num)+" unavailable")
        return None
    return bscan_name

def getHypotheneuse(coordinate, start_coordinate, end_coordinate):
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
    if segment is None:
        print("Segment not available at ")
        print(coordinate)
        return None
    return segment


def circle(x, y, sigma=9.0589887): # 6.0393258 + 0.5
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

# def circle(x, y, sigma=6.0393258 + 0.5, step=4):
#     """
#     Create a small collection of points in a neighborhood of some point.
#     The number of points can be controlled by adjusting the step size.
#     """
#     # 2: 39; 3: 17; 4: 12
#     neighborhood = []
#
#     X = int(sigma)
#     for i in range(-X, X + 1, step):  # Adjust step size to reduce the number of points
#         Y = int(np.sqrt(sigma * sigma - i * i))
#         for j in range(-Y, Y + 1, step):  # Adjust step size to reduce the number of points
#             neighborhood.append((x + i, y + j))
#
#     return neighborhood

# def circle(x, y, sigma=9.0589887 + 0.5): # 6.0393258
#     """
#     Create a list of 9 points:
#     - Point 1: The center point (x, y)
#     - Points 2-5: The points (x, y-sigma), (x-sigma, y), (x+sigma, y), (x, y+sigma)
#     - Points 6-9: The points (x-sigma/2, y-sigma/2), (x-sigma/2, y+sigma/2), (x+sigma/2, y-sigma/2), (x+sigma/2, y+sigma/2)
#     """
#     neighborhood = [
#         (x, y),  # Center point
#         (x, y - sigma),  # Top point
#         (x - sigma, y),  # Left point
#         (x + sigma, y),  # Right point
#         (x, y + sigma),  # Bottom point
#         (x - sigma / 2, y - sigma / 2),  # Top-left point
#         (x - sigma / 2, y + sigma / 2),  # Bottom-left point
#         (x + sigma / 2, y - sigma / 2),  # Top-right point
#         (x + sigma / 2, y + sigma / 2)   # Bottom-right point
#     ]
#
#     return neighborhood


# def noOverlap(x, y, sigma=13):
#     return [(x,y),(x,y+sigma/4),(x,y-sigma/4),(x,y+sigma/2),(x,y-sigma/2),(x,y-sigma/4*3),(x,y-sigma/4*3),(x,y-sigma),(x,y-sigma)]
# #len(noOverlap(355,367.89,13))


# edited: added ->
# def proportional_distance(a, b, c):
#     """
#     Calculates the proportional distance from point c to the line connecting a and b.
#     """
#     # Calculate the length of the line segment
#     line_length = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
#
#     # Calculate the perpendicular distance, normalized by line length
#     distance = abs(c[1] - a[1]) / line_length
#
#     return distance

def is_closed_enough(a,b,c, closeCP = 2048): # or crossProd < 600
    # prop_dist = abs(proportional_distance(a, b, c))
    # close_standard = 0.000572

    # when step=4
    # 600: each image has 7500 slices created
    # 0.0512: each image has 6500 slices created
    crossP = abs(crossProd(a,b,c))
    if crossP < closeCP: # prop_dist < close_standard:
        return True
    return False


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
# <-
###############3

train_num=0
train_names=[]
train_labels=[]
val_num=0
val_names=[]
val_labels=[]
test_num=0
test_names=[]
test_labels=[]

# edited: added ->
diameter_in_um = 129
radius_in_um = diameter_in_um/2
radius_in_px = radius_in_um/10.68
R = radius_in_px
# <-


for k in range(MPinfo.shape[0]):  # each image, (every row?)
    oct_dir = MPinfo.loc[k][4]
    # column 4: E2E_path, eg: Visit_1_extracted/019_OS.E2E_extracted

    # Use stretched_SLO_coordinates_0.csv if there's one # edited
    # oct_info = genfromtxt(oct_dir+'/SLO_coordinates_0.csv', delimiter=',', dtype=int, encoding='utf-8-sig')[1:,]
    potential_stretched_slo_coord_path = oct_dir+'/stretched_SLO_coordinates_0.csv'
    if os.path.exists(potential_stretched_slo_coord_path) and os.path.isfile(potential_stretched_slo_coord_path):
        oct_info = genfromtxt(oct_dir+'/stretched_SLO_coordinates_0.csv', delimiter=',', dtype=int, encoding='utf-8-sig')[1:,]
    else:
        oct_info = genfromtxt(oct_dir+'/SLO_coordinates_0.csv', delimiter=',', dtype=int, encoding='utf-8-sig')[1:,]

    oct_coords = np.apply_along_axis(formatCoords, 1, oct_info)
    mp_id = MPinfo.loc[k][0]  # e.g.: 57_234

    # Use stretched_MP_mpid.csv if there's one # edited
    # mp_points = genfromtxt('Registered_coords/MP_'+mp_id+'.csv', dtype=float, delimiter=',', encoding='utf-8-sig')
    potential_stretched_mp_coord_path = 'Registered_coords/stretched_MP_'+mp_id+'.csv'
    if os.path.exists(potential_stretched_mp_coord_path) and os.path.isfile(potential_stretched_mp_coord_path):
        mp_points = genfromtxt('Registered_coords/stretched_MP_'+mp_id+'.csv', dtype=float, delimiter=',', encoding='utf-8-sig')
    else:
        mp_points = genfromtxt('Registered_coords/MP_'+mp_id+'.csv', dtype=float, delimiter=',', encoding='utf-8-sig')

    mp_values = MPvalues[:,k]
    assert(len(mp_points) == len(mp_values))
    print(oct_dir)

    # edited ->
    for j, mp_centre in enumerate(mp_points): # each point
        # skip the point if mp value is -1
        if mp_values[j] == -1:
            continue
        # total_slices_for_that_mppoint = 0 ## count
        bscan_used_for_that_mppoint = [] ## count
        oversample = circle(mp_centre[0], mp_centre[1])
        # print(oversample)
        for m, mp_coords in enumerate(oversample):
            for i, oct_coord in enumerate(oct_coords):
                #print(oct_coord)
                start_coords = oct_coord[0]
                end_coords = oct_coord[1]
                # Skip such Bscan line if start = end
                if start_coords[0] == end_coords[0]:
                    # print(oct_dir+str(i)+" start coords == end_coords")
                    continue
                # assert(start_coords[0] != end_coords[0])
                # Skip such point if not closed enough with bscan line
                if not is_closed_enough(start_coords, end_coords, mp_coords):
                    continue
                bscan_name = getBscan(i)

                # print(oct_coord)

                # which_volume_folder = '/normalized_volume_0/'  # oct_dir = E2E_path here
                which_volume_folder = '/volume_0/'  # oct_dir = E2E_path here
                if os.path.exists(oct_dir + '/stretched_volume_0') and \
                        os.path.isdir(oct_dir + '/stretched_volume_0'):
                    which_volume_folder = '/stretched_volume_0/'
                bscan_path = oct_dir + which_volume_folder + bscan_name

                # which dataset
                save_dir = save_directory(bscan_path=bscan_path)

                bscan = cv2.imread(bscan_path)  # read bscan .png
                coord = getHypotheneuse(mp_coords, start_coords, end_coords) # assuming OCT scans are horizontal
                bscan_slice = getSeg(coord, bscan)

                # skip if segment unavailable
                if bscan_slice is None or bscan_slice.size == 0:
                    print("bscan_slice is None or bscan_slice.size == 0 at:")
                    print(bscan_path)
                    print(coord)
                    continue

                # do different sampling based on the datasets
                if save_dir == 'Train/train':
                    cv2.imwrite(save_dir+str(train_num)+'.png', bscan_slice) ##

                    train_names.append(save_dir+str(train_num)+'.png')
                    train_labels.append(mp_values[j])
                    train_num += 1
                    # pass
                elif save_dir == 'Validate/val':
                    cv2.imwrite(save_dir+str(val_num)+'.png', bscan_slice)

                    val_names.append(save_dir+str(val_num)+'.png')
                    val_labels.append(mp_values[j])
                    val_num += 1
                    # pass
                else:
                    assert save_dir == 'Test/test'
                    cv2.imwrite(save_dir+str(test_num)+'.png', bscan_slice)

                    test_names.append(save_dir+str(test_num)+'.png')
                    test_labels.append(mp_values[j])
                    test_num += 1
                    # pass
                if bscan_name not in bscan_used_for_that_mppoint:
                    bscan_used_for_that_mppoint.append(bscan_name)
        if len(bscan_used_for_that_mppoint) == 0: ## count
            print("start each mp point:")
            print(mp_centre[0], mp_centre[1])
            print("no slice generated by the mp point ")
        #
        #     # print(mp_centre[0], mp_centre[1])
        # elif len(bscan_used_for_that_mppoint) == 1:
        #     # print(bscan_used_for_that_mppoint)
        #     continue
        # else:
        #     print("start each mp point:")
        #     print(mp_centre[0], mp_centre[1])
        #     print(len(bscan_used_for_that_mppoint))
        #     print(bscan_used_for_that_mppoint)

        # else: ## count
        #     # print("total_slices_for_that_mppoint != 9")
        #     # print(mp_centre[0], mp_centre[1])
        #     print(total_slices_for_that_mppoint)

                #
                # which_dataset_names.append(save_dir+str(which_dataset_num)+'.png')
                # which_dataset_labels.append(mp_values[j])
                # which_dataset_num += 1

                # bscan = cv2.imread(oct_dir + which_volume_folder + bscan_name)
                # coord = getHypotheneuse(mp_coords, start_coords, end_coords) # assuming OCT scans are horizontal
                # bscan_slice = getSeg(coord, bscan)
                # cv2.imwrite(save_dir+str(train_num)+'.png', bscan_slice)
                # train_names.append(save_dir+str(train_num)+'.png')
                # train_labels.append(mp_values[j])
                # train_num += 1
        # print(mp_centre[0], mp_centre[1])
    print(train_num, val_num, test_num)
    print(time.time()-start_time)
    # <-


# edited: added ->
# check the length of train, val, test - slices
print("amount of slices in train, val and test sets:")
print(train_num)
print(val_num)
print(test_num)
print("\\")
total_slices = train_num + val_num + test_num
print("ratio of slices in train/all, validation/all, test/all")
print(float(train_num/total_slices))
print(float(val_num/total_slices))
print(float(test_num/total_slices))
#  <-


with open('train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(train_names, train_labels))

with open('val.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(val_names, val_labels))

with open('test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_names, test_labels))

