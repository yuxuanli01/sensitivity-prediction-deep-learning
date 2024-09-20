# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # # Load the data
# # train_df = pd.read_csv('train.csv', header=None)
# # val_df = pd.read_csv('val.csv', header=None)
# # test_df = pd.read_csv('test.csv', header=None)
# #
# # # Extract sensitivity values
# # train_sensitivity = train_df.iloc[:, 1]
# # val_sensitivity = val_df.iloc[:, 1]
# # test_sensitivity = test_df.iloc[:, 1]
# #
# # # Plotting histograms
# # plt.figure(figsize=(15, 5))
# #
# # # Train set
# # plt.subplot(1, 3, 1)
# # plt.hist(train_sensitivity, bins=range(int(train_sensitivity.min()), int(train_sensitivity.max()) + 1), edgecolor='black')
# # plt.title('Train Set Sensitivity Distribution')
# # plt.xlabel('Sensitivity (dB)')
# # plt.ylabel('Frequency')
# #
# # # Validation set
# # plt.subplot(1, 3, 2)
# # plt.hist(val_sensitivity, bins=range(int(val_sensitivity.min()), int(val_sensitivity.max()) + 1), edgecolor='black')
# # plt.title('Validation Set Sensitivity Distribution')
# # plt.xlabel('Sensitivity (dB)')
# # plt.ylabel('Frequency')
# #
# # # Test set
# # plt.subplot(1, 3, 3)
# # plt.hist(test_sensitivity, bins=range(int(test_sensitivity.min()), int(test_sensitivity.max()) + 1), edgecolor='black')
# # plt.title('Test Set Sensitivity Distribution')
# # plt.xlabel('Sensitivity (dB)')
# # plt.ylabel('Frequency')
# #
# # # Display the plots
# # plt.tight_layout()
# # plt.show()
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the data
# train_df = pd.read_csv('train.csv', header=None)
# val_df = pd.read_csv('val.csv', header=None)
# test_df = pd.read_csv('test.csv', header=None)
#
# # Extract sensitivity values
# train_sensitivity = train_df.iloc[:, 1]
# val_sensitivity = val_df.iloc[:, 1]
# test_sensitivity = test_df.iloc[:, 1]
#
# #
# # # Combine all the sensitivity values into a single array
# # all_sensitivity = pd.concat([train_sensitivity, val_sensitivity, test_sensitivity])
# #
# # # Plotting the combined distribution
# # plt.figure(figsize=(10, 6))
# #
# # # Plot combined sensitivity distribution
# # plt.hist(all_sensitivity, bins=30, alpha=0.7, density=True, color='blue')
# #
# # # Labels and title
# # plt.title('Combined Sensitivity Distribution Across All Datasets')
# # plt.xlabel('Sensitivity')
# # plt.ylabel('Ratio of Frequency')
# #
# # plt.show()
# # #
# # # # Plotting histograms with rate of frequency (relative frequency)
# # # plt.figure(figsize=(15, 5))
# # #
# # # # Train set
# # # plt.subplot(1, 3, 1)
# # # plt.hist(train_sensitivity, bins=range(int(train_sensitivity.min()), int(train_sensitivity.max()) + 1),
# # #          edgecolor='black', density=True)
# # # plt.title('Train Set Sensitivity Distribution')
# # # plt.xlabel('Sensitivity (dB)')
# # # plt.ylabel('Rate of Frequency')
# # #
# # # # Validation set
# # # plt.subplot(1, 3, 2)
# # # plt.hist(val_sensitivity, bins=range(int(val_sensitivity.min()), int(val_sensitivity.max()) + 1),
# # #          edgecolor='black', density=True)
# # # plt.title('Validation Set Sensitivity Distribution')
# # # plt.xlabel('Sensitivity (dB)')
# # # plt.ylabel('Rate of Frequency')
# # #
# # # # Test set
# # # plt.subplot(1, 3, 3)
# # # plt.hist(test_sensitivity, bins=range(int(test_sensitivity.min()), int(test_sensitivity.max()) + 1),
# # #          edgecolor='black', density=True)
# # # plt.title('Test Set Sensitivity Distribution')
# # # plt.xlabel('Sensitivity (dB)')
# # # plt.ylabel('Rate of Frequency')
# # #
# # # # Display the plots
# # # plt.tight_layout()
# # # plt.show()
#
# # Plotting the distribution for the training set only
# plt.figure(figsize=(10, 6))
#
# # Plot train sensitivity distribution
# plt.hist(train_sensitivity, bins=12, alpha=0.7, density=True, color='green')
#
# # Labels and title
# plt.title('Sensitivity Distribution')
# plt.xlabel('Sensitivity')
# plt.ylabel('Ratio of Frequency')
#
# plt.show()



# training loss:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('resnet18_t1_epoch12.csv') # resnet18_settings1_10.csv

# Convert the losses into dB and ensure proper conversion to NumPy arrays
df['Training Loss (dB)'] = 10 * np.log10(df['Training Loss'].to_numpy())
df['Validation Loss (dB)'] = 10 * np.log10(df['Validation Loss'].to_numpy())
df['Validation L1 Loss (dB)'] = 10 * np.log10(df['Validation L1 Loss'].to_numpy())

# Plot the training loss and validation loss curves in dB
plt.figure(figsize=(10, 6))

# Convert the 'Epoch' column to a NumPy array to avoid indexing issues
epochs = df['Epoch'].to_numpy()

# Plot Training Loss in dB
plt.plot(epochs, df['Training Loss (dB)'].to_numpy(), label='Training Loss (dB)', color='blue', marker='o')

# Plot Validation Loss in dB
# plt.plot(epochs, df['Validation Loss (dB)'].to_numpy(), label='Validation Loss (dB)', color='red', marker='o')

# Plot Validation L1 Loss (MAE) in dB
plt.plot(epochs, df['Validation L1 Loss (dB)'].to_numpy(), label='Validation L1 Loss (dB)', color='green', marker='o')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss (dB)')
plt.title('Training and Validation Loss in dB')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

### normal scale:
#
# # Plot the training loss and validation loss curves in normal scale
plt.figure(figsize=(10, 6))

# Convert the 'Epoch' column to a NumPy array to avoid indexing issues
epochs = df['Epoch'].to_numpy()

# Plot Training Loss in normal scale
plt.plot(epochs, df['Training Loss'].to_numpy(), label='Training Loss', color='blue', marker='o')

# Plot Validation Loss in normal scale
# plt.plot(epochs, df['Validation Loss'].to_numpy(), label='Validation Loss', color='red', marker='o')

# Plot Validation L1 Loss (MAE) in normal scale
plt.plot(epochs, df['Validation L1 Loss'].to_numpy(), label='Validation L1 Loss', color='green', marker='o')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss (Normal Scale)')
plt.title('Training and Validation Loss in Normal Scale')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()




#
# #
# # import pandas as pd
# # import glob
# # import os
# #
# # # Define the root directory containing all subfolders with CSV files
# # root_dir = ['Visit_1', 'Visit_2']
# #
# # csv_files = []
# #
# # for dir in root_dir:
# #     for subfolder in os.listdir(dir):
# #         eye_path = os.path.join(dir,subfolder,'volume_0')
# #         if not os.path.isdir(eye_path):
# #             continue
# #         for file in os.listdir(eye_path):
# #             if file.endswith('.csv'):
# #                 file_path = os.path.join(eye_path, file)
# #                 csv_files.append(file_path)
# #
# # # # Recursively find all CSV files in the root directory and its subdirectories
# # # csv_files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
# #
# # # Read and combine all CSV files
# # all_data = pd.concat([pd.read_csv(f, sep='\t') for f in csv_files], ignore_index=True)
# #
# # # Calculate mean, standard deviation, and range for each structure
# # statistics = all_data.describe().loc[['mean', 'std', 'min', 'max']]
# # statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
# #
# # # Drop unnecessary columns (like 'x' if not needed)
# # statistics = statistics.drop(columns=['x'])
# #
# # # Display the result
# # print(statistics)
# #
#
# import pandas as pd
# import os
#
# # Define the root directories containing all subfolders with CSV files
# root_dirs = ['Visit_1', 'Visit_2']
#
# csv_files = []
#
# # Collect all CSV files in specified directories
# for root_dir in root_dirs:
#     for subfolder in os.listdir(root_dir):
#         eye_path = os.path.join(root_dir, subfolder, 'volume_0')
#         if not os.path.isdir(eye_path):
#             continue
#         for file in os.listdir(eye_path):
#             if file.endswith('.csv'):
#                 file_path = os.path.join(eye_path, file)
#                 csv_files.append(file_path)
#
# # Read and combine all CSV files, ensuring proper data types
# all_data_list = []
# for file in csv_files:
#     # Adjust separator if necessary (e.g., use sep=',' if files are comma-separated)
#     df = pd.read_csv(file, sep=',')  # Use ',' if the files are comma-separated
#     # Convert columns (except 'x') to numeric, removing non-numeric characters if needed
#     for column in df.columns:
#         if column != 'x':  # Skip 'x' column
#             # Remove any non-numeric characters and convert to numeric
#             df[column] = pd.to_numeric(df[column].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')
#     all_data_list.append(df)
#
# # Combine all data into a single DataFrame
# all_data = pd.concat(all_data_list, ignore_index=True)
#
# # Inspect the data to ensure it's read correctly
# print(all_data.head())  # Display the first few rows
# print(all_data.dtypes)  # Check the data types
#
# # Calculate mean, standard deviation, and range for each structure
# statistics = all_data.describe().loc[['mean', 'std', 'min', 'max']]
# statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
#
# # Drop unnecessary columns (like 'x' if not needed)
# statistics = statistics.drop(columns=['x'], errors='ignore')
#
# # Display the result
# print(statistics)
