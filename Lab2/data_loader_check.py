# data_loader_check.py

from torch.utils.data import DataLoader
from dataset import PetNoseDataset # Imports the custom dataset class defined in dataset.py
import torchvision.transforms as T # Imports transformation utilities like ToTensor()
import matplotlib.pyplot as plt # For visualizing images and coordinates (nose position)
import argparse # Allows for command-line argument parsing

# Define a command-line argument parser to accept arguments for image directory, labels file, and number of samples to visualize
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to images directory') # Argument for the images directory path
    parser.add_argument('--labels_file', type=str, help='Path to labels file') # Argument for the labels file path
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to check') # Argument for the number of samples to visualize, default is 10
    args = parser.parse_args() # read the command-line arguments that were provided when running the script then parse and assign these arguments to the args object

    # Define the transformation to apply to the images
    transform = T.Compose([
        T.ToTensor(),
    ]) # Converts the image to a PyTorch tensor (scales pixel values to [0, 1])
    
    # Create the dataset using the custom PetNoseDataset class, applying the transformation and no augmentations
    dataset = PetNoseDataset(
        images_dir=args.data_dir, 
        labels_file=args.labels_file, 
        transform=transform, 
        augmentations=None
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Create the DataLoader to iterate through the dataset in batches of size 16, shuffling the dataset

    for idx, (image, target) in enumerate(dataloader): # Iterate through the dataset, loading one image and target at a time
        print(f"Sample {idx+1}:") # Print out the sample number, image shape, and target coordinates
        print(f"Image shape: {image.shape}") # Prints the shape of the image tensor
        print(f"Target coordinates: {target}") # Prints the coordinates of the nose (target)
        
        # Visualize the image and the target nose position
        img = image.squeeze(0).permute(1, 2, 0).numpy() # Remove batch dimension, rearrange channels for plotting, convert to NumPy array
        img = img.clip(0, 1)  # Ensure pixel values are between 0 and 1 for display
        plt.imshow(img) # Display the image

        u, v = target[0] # Extract the coordinates from the target tensor
        # Convert tensors to Python scalars
        u = u.item()
        v = v.item()
        plt.scatter(u, v, color='red') # Plot the nose position as a red dot on the image
        plt.title(f"Sample {idx+1}") # Set the title of the plot
        plt.show() # Display the plot
        
        # Stop after visualizing the specified number of samples (from the argument `num_samples`)
        if idx+1 >= args.num_samples:
            break

# Expected Output
# For each sample, the script will display:

# Sample 1:
# Image shape: torch.Size([1, 3, 227, 227])
# Target coordinates: tensor([[value1, value2])
# Sample 2: ...

# It will also display the image with a red dot indicating the ground truth nose position

if __name__ == '__main__':
    # Stop after visualizing the specified number of samples (from the argument `num_samples`)
    main()