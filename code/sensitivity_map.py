import numpy as np
from PIL import Image
import pandas as pd
import os
from loader_new4 import eval_transform
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def cut_oct_image_into_slices(image_path, slice_width=32, output_dir='35OD2_slices'):
    """
    Cut an OCT image into vertical slices of given width and save them to a directory.

    Parameters:
        image_path (str): Path to the OCT image file.
        slice_width (int): Width of each vertical slice.
        output_dir (str): Base directory where slices will be saved.

    Returns:
        None: Saves slices as image files in a subdirectory.
    """
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Create a subdirectory based on the image file name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    subfolder_path = os.path.join(output_dir, image_name)
    os.makedirs(subfolder_path, exist_ok=True)

    height, width = image_np.shape
    slice_count = 0

    # Cut the image into slices and save them
    for i in range(0, width - slice_width + 1, slice_width):
        slice_ = image_np[:, i:i + slice_width]
        slice_image = Image.fromarray(slice_)
        slice_filename = os.path.join(subfolder_path, f'slice_{slice_count}.png')
        slice_image.save(slice_filename)
        slice_count += 1
        print(f'Saved {slice_filename}')

    print(f'All slices saved to {subfolder_path}')

image_path_test000='Visit_2_extracted/035_OD.E2E_extracted/volume_0/volume_000.png'
cut_oct_image_into_slices(image_path_test000)

def cut_oct_image_into_slices(image_path, slice_width=32, output_dir='35OD2_slices'):
    """
    Cut an OCT image into vertical slices of given width and save them to a directory.

    Parameters:
        image_path (str): Path to the OCT image file.
        slice_width (int): Width of each vertical slice.
        output_dir (str): Base directory where slices will be saved.

    Returns:
        None: Saves slices as image files with names based on the original image name.
    """
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Extract image name (without extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    height, width = image_np.shape
    slice_count = 0

    # Cut the image into slices and save them
    for i in range(0, width - slice_width + 1, slice_width):
        slice_ = image_np[:, i:i + slice_width]
        slice_image = Image.fromarray(slice_)
        slice_filename = os.path.join(output_dir, f'{image_name}_slice{slice_count}.png')
        slice_image.save(slice_filename)
        slice_count += 1
        print(f'Saved {slice_filename}')

    print(f'All slices saved to {output_dir}')

for num in range(0, 49):  # Loop from 0 to 48
    path = f"Visit_2_extracted/035_OD.E2E_extracted/volume_0/volume_{num:03d}.png"  # Format number with leading zeros, e.g., 001, 002, ..., 048
    cut_oct_image_into_slices(path)


class OCTSlicesDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(base_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.base_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, self.image_filenames[idx]  # Return the filename along with the image


def evaluate_model(model, data_loader):
    all_predictions = []
    all_filenames = []

    model.eval()
    with torch.no_grad():
        for inputs, filenames in data_loader:
            outputs = model(inputs)
            predictions = outputs.numpy().flatten()  # Convert to NumPy array and flatten if needed
            all_predictions.extend(predictions)
            all_filenames.extend(filenames)
    return all_filenames, all_predictions

def save_predictions_to_csv(filenames, predictions, output_csv='35OD2_predictions.csv'):
    # Create a DataFrame
    df = pd.DataFrame({
        'Filename': filenames,
        'Prediction': predictions
    })
    # Save to CSV
    df.to_csv(output_csv, index=False)


model=torch.load('resnet18_t1_epoch12.pt')
# model.eval()

dataset = OCTSlicesDataset(base_dir='35OD2_slices', transform=eval_transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# Evaluate the model
filenames, predictions = evaluate_model(model, data_loader)

# Save the results to a CSV file
save_predictions_to_csv(filenames, predictions)
