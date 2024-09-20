import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def calculate_slope(pt1, pt2):
    if pt2[0] - pt1[0] == 0:
        return float('inf')  # handle division by zero
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def filter_matches_by_slope(matches, kp1, kp2, slope_threshold_factor=2):
    slopes = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        slope = calculate_slope(pt1, pt2)
        slopes.append(slope)

    slopes = np.array(slopes)
    median_slope = np.median(slopes)
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    min_slope = median_slope - slope_threshold_factor * iqr
    max_slope = median_slope + slope_threshold_factor * iqr

    filtered_matches = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        slope = calculate_slope(pt1, pt2)
        if min_slope <= slope <= max_slope:
            filtered_matches.append(match)

    return filtered_matches

def imgRegister(query_img, train_img):
    '''
    query_img: mp slo
    train_img: oct slo
    '''
    dim = (1536, 1536)
    if train_img.shape == (1024, 1024, 3):
        train_img = cv2.resize(train_img, dim)
    # Convert to grayscale
    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    print(img2.shape)


    ### Feature matching ###

    # Create ORB detector with 5000 features
    orb_detector = cv2.ORB_create(nfeatures=5000)

    # Detect keypoints and compute descriptors
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)

    # Sort matches based on their Hamming distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter matches based on the slopes of the lines connected by matched keypoints
    matches = filter_matches_by_slope(matches, kp1, kp2)
    no_of_matches = len(matches)

    # Draw matches between the two images
    feat_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)


    ### Image registration ###

    # Initialize matrices to hold matched keypoints' coordinates
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Estimate the optimal 2D affine transformation between the point sets
    affine_mat, _ = cv2.estimateAffine2D(p1, p2)

    # Warp the query image to align with the reference image using the affine transformation
    aligned_img = cv2.warpAffine(query_img, affine_mat, (train_img.shape[1], train_img.shape[0]))
    print(aligned_img.shape)

    # Create an overlay of the registered image on the reference image for visualization
    overlay = train_img.copy()
    output = aligned_img.copy()
    cv2.addWeighted(output, 0.5, overlay, 0.5, 0, overlay)

    return aligned_img, overlay, affine_mat, feat_img, img1, img2

# Load CSV file containing image information
IDs = pd.read_csv('OCT_MP_match.csv')

dir_store = []
def getImages(row):
    slo_no = row[0]
    patient = row[1]
    eye = row[2]
    exam1 = row[3]
    exam2 = row[4]

    if slo_no == 39:
        return

    if 19 <= slo_no <= 33:
        oct_dir1 = f'Visit_1_extracted/0{slo_no:02d}_{eye}.E2E_extracted/images/2D_0.png'
    elif 34 <= slo_no <= 55:
        oct_dir1 = f'Visit_1_extracted/0{slo_no:02d}_{eye}_49.E2E_extracted/images/2D_0.png'
    else:
        raise ValueError(f"Unexpected slo_no: {slo_no}")

    mp_dir1 = f'USH_MP_clean/MAIA_exam_{patient}_{exam1}.png'
    dir_store.append((mp_dir1, oct_dir1))

    if slo_no != 51:
        oct_dir2 = f'Visit_2_extracted/0{slo_no:02d}_{eye}.E2E_extracted/images/2D_0.png'
        mp_dir2 = f'USH_MP_clean/MAIA_exam_{patient}_{exam2}.png'
        dir_store.append((mp_dir2, oct_dir2))

# Apply getImages function to each row in the CSV
IDs.apply(getImages, axis=1)

# Perform image registration for each pair of images
for i, pair in enumerate(dir_store):
    mp_image = cv2.imread(pair[0])
    oct_image = cv2.imread(pair[1])
    reg_img, overlay, affine_mat, feat_img, img1, img2 = imgRegister(mp_image, oct_image)

    print(pair)
    img_id = pair[0][22:]
    cv2.imwrite('Registered/registered' + img_id, reg_img)
    cv2.imwrite('Registered/overlay' + img_id, overlay)
    cv2.imwrite('Registered/feat' + img_id, feat_img)
    np.savetxt('Registered/matrix' + img_id[:-3] + 'csv', affine_mat, delimiter=',')

