import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


def loadcsv_updatethickness(path, volume_number, thickness_table):
    df = pd.read_csv(path)
    df['thickness'] = 4.2*(df['IBRPE'] - df['OPL'])
    newlist = df['thickness'].tolist()
    for each in newlist:
        thickness_table[volume_number].append(each)


# root_dir = 'Visit_1' or 'Visit_2'
def root_dir_path(path: 1 or 2):
    return 'Visit_1' if path == 1 else 'Visit_2'

for p in [1, 2]:  # 1 completed
    root_dir = root_dir_path(p)

    # Iterate through each subfolder in the root directory
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        # Check if the current path is a directory
        if os.path.isdir(subfolder_path):
            directory = os.path.join(subfolder_path, 'volume_0')
            csv_files = glob.glob(os.path.join(directory, '*.csv'))
            thickness_table = [[] for _ in range(len(csv_files))]  #thickness table for each folder

            # Find all CSV files in the current subfolder and update thickness table
            for i in range(len(csv_files)):
                file_path = os.path.join(subfolder_path, f'volume_0/volume_{i:03d}.csv')

                # Check if the file exists
                if os.path.exists(file_path):
                    loadcsv_updatethickness(file_path, i, thickness_table)
            print("generating " + str(root_dir) + ": " + subfolder[:6] + '_thicknessmap.png')
            # Convert the thickness_table list to a 2D NumPy array
            thickness_array = np.array(thickness_table)

            # Generate the plot
            plt.figure(figsize=(10, 10))
            plt.imshow(thickness_array, cmap='viridis', aspect='auto')
            plt.colorbar(label='Retinal thickness (µm)')
            plt.title('Retinal Thickness Map')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')

            # Save the plot to the corresponding folder
            image_name = subfolder[:6] + '_thicknessmap.png'
            image_path = os.path.join(subfolder_path, image_name)
            plt.savefig(image_path)
            plt.close()  # Close the plot to free memory

