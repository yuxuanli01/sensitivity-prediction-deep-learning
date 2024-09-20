import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy import genfromtxt
import cv2
import csv
import os
from PIL import Image



def is_normalized(image_path):
    """
    Check if an image has been 0-1 normalized.

    Parameters:
    - image: A numpy array representing the image.

    Returns:
    - True if the image is normalized (i.e., all pixel values are between 0 and 1 inclusive).
    - False otherwise.
    """
    imageopen = Image.open(image_path)

    # Convert image to numpy array
    image = np.array(imageopen)
    return np.min(image) >= 0.0 and np.max(image) <= 1.0


def normalize01(imagepath, newpath):
    image = Image.open(imagepath)

    # Convert image to numpy array
    image_array = np.array(image)

    # Normalize the image to 0-1 range
    normalized_image_array = image_array / 255.0

    # Optionally, if you want to save the normalized image, scale it back to 0-255
    rescaled_image_array = (normalized_image_array).astype(np.uint8)

    # Convert back to PIL Image and save
    rescaled_image = Image.fromarray(rescaled_image_array)
    rescaled_image.save(newpath)

    print("Image " + imagepath + " has been normalized and saved.")


visit_1_dir = 'Visit_1_extracted'
visit_2_dir = 'Visit_2_extracted'

# List of the two directories to loop through
directories = [visit_1_dir, visit_2_dir]

# Loop over each directory
for parent_dir in directories:
    # Loop over each folder within the current parent directory
    for E2E_folder in os.listdir(parent_dir):
        E2E_path = os.path.join(parent_dir, E2E_folder)
        if not os.path.isdir(E2E_path):  # Skip if it's not a directory
            continue

        # Skip if there's a folder normalized_volume_0 inside E2E_path
        if os.path.isdir(os.path.join(E2E_path, 'normalized_volume_0')):
            continue
        # Create normalized folder
        norm_dir = os.path.join(E2E_path, 'normalized_volume_0')
        os.makedirs(norm_dir)
        print(f"Folder created at: {norm_dir}")
        if os.path.isdir(os.path.join(E2E_path, 'stretched_volume_0')):
            original_path = os.path.join(E2E_path, 'stretched_volume_0')
        else:
            original_path = os.path.join(E2E_path, 'volume_0')
        for volumes in os.listdir(original_path):
            volume_path = os.path.join(original_path, volumes)
            print(volume_path)
            if not is_normalized(volume_path):
                normalize01(volume_path, os.path.join(norm_dir, volumes))
