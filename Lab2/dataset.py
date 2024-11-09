# dataset.py

import torch
from torch.utils.data import Dataset  # Import PyTorch's Dataset class to create custom datasets
import os # Import OS library for file path manipulations.
from PIL import Image # Import Python Imaging Library (PIL) to handle image loading and processing

# Define a custom dataset class `PetNoseDataset` which inherits from PyTorch's `Dataset`.
class PetNoseDataset(Dataset): 
    def __init__(self, images_dir, labels_file, transform=None, augmentations=None):
        """
        Initialization function for the dataset.

        Args:
            images_dir (string): Directory with all the images.
            labels_file (string): Path to the file that contains the image labels (the nose coordinates).
            transform (callable, optional): Transform to be applied on the PIL Image (like converting to tensor).
            augmentations (callable, optional): Augmentations to be applied on both image and coordinates.
        """
        self.images_dir = images_dir # Store the directory containing the images
        self.labels = []  # Initialize an empty list to store image names and nose coordinates
        self.transform = transform # Store the image transformation function
        self.augmentations = augmentations # Store the augmentations function
        
        # Read the labels file and extract image names and their corresponding nose coordinates
        with open(labels_file, 'r') as f:
            lines = f.readlines() # Read all lines from the labels file
            for line in lines: # For each line: image_name,"(u, v)"
                parts = line.strip().split(',', 1) # Split each line into the image name and the coordinates
                image_name = parts[0] # Extract the image name
                coords_str = ','.join(parts[1:]).strip('"()') # Extract the u, v coordinates (stripping quotes and parentheses)
                u_str, v_str = coords_str.split(',') # Split the coordinates string into u and v values
                u = float(u_str)  # Convert the u-coordinate from string to float
                v = float(v_str)
                self.labels.append((image_name, (u, v)))  # Append the image name and (u, v) coordinates as a tuple to the labels list
    
    def __len__(self): 
    # The `__len__` function returns the total number of items in the dataset
        return len(self.labels) # The length is the number of label entries
    
    def __getitem__(self, idx):
    # The `__getitem__` function retrieves a sample (image and target coordinates) by index
        image_name, coords = self.labels[idx] # Extract the image name and nose coordinates from the labels list
        img_path = os.path.join(self.images_dir, image_name) # Create the full image path by combining the image directory with the image name
        image = Image.open(img_path).convert('RGB') # Load the image using PIL and convert it to RGB format
        u, v = coords # Extract the u, v coordinates
        
        # Apply data augmentations (if any) to both the image and coordinates
        if self.augmentations:
            image, (u, v) = self.augmentations(image, (u, v))
        
        orig_width, orig_height = image.size # Get the original image width and height after any augmentations
        
        image = image.resize((227, 227)) # Resize the image to 227x227 (the input size expected by the model)
        # Calculate scaling factors for the coordinates (based on the original image dimensions)
        scale_x = 227 / orig_width
        scale_y = 227 / orig_height
        # Adjust the u and v coordinates according to the scaling factors
        u = u * scale_x
        v = v * scale_y
        
        # Apply image transformations (e.g., converting the image to a tensor)
        if self.transform:
            image = self.transform(image)
        
        target = torch.tensor([u, v], dtype=torch.float32) # Prepare the target tensor containing the adjusted u, v coordinates
        
        return image, target # Return the processed image and its corresponding target coordinates