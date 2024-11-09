# test.py

import torch
from torch.utils.data import DataLoader
from model import SnoutNet
from dataset import PetNoseDataset
import torchvision.transforms as T
import argparse
import time
import numpy as np # Imports numpy for numerical operations
import matplotlib.pyplot as plt
import os

# Argument parser to handle command-line arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='Path to images directory')
    parser.add_argument('-l', '--test_labels', type=str, help='Path to test labels file')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the trained model')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs', help='Directory to save output images')
    args = parser.parse_args() # Parses the command-line arguments into 'args'

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Defines the device to run the model on (GPU if available, otherwise CPU
    
    # Define the data transformations
    transform = T.Compose([ # Compose a set of transformations
        T.ToTensor() # Converts the images to tensors, which are required by PyTorch models
    ])
    
    # Create the dataset and dataloader for testing
    test_dataset = PetNoseDataset( # Instantiates the custom dataset class for testing
        images_dir=args.data_dir,  # Uses the provided image directory
        labels_file=args.test_labels, # Uses the provided test labels file
        transform=transform, # Apply the defined transformation (ToTensor)
        augmentations=None  # No augmentations are applied during testing
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # Loads the data with no shuffling (testing doesn't need shuffled data)
    
    # Initialize the model
    model = SnoutNet().to(device) # Initializes the model and sends it to the specified device (CPU/GPU)
    model.load_state_dict(torch.load(args.model_path, map_location=device)) # Loads the trained model's state from the specified file
    model.eval()  # Sets the model to evaluation mode (disables dropout and batch normalization effects)
    
    # List to store error values, inference times, images, model's predictions, and ground truth (actual) coordinates for each image
    errors = []
    times = []
    images_list = [] 
    predictions = [] 
    ground_truths = [] 

    os.makedirs(args.output_dir, exist_ok=True) # Create the output directory if it doesn't exist
    
    with torch.no_grad(): # Disables gradient calculation during inference (faster and saves memory)
        for images, targets in test_loader: # Iterates over the test dataset in batches
            images = images.to(device) # Sends the input images to the specified device (CPU/GPU).
            targets = targets.to(device) # Sends the ground-truth labels (targets) to the specified device (CPU/GPU)
            start_time = time.time() # Start measuring inference time
            outputs = model(images) # Forward pass through the model to get predictions
            end_time = time.time()  # Stop measuring inference time
            inference_time = (end_time - start_time) * 1000  # Converts the inference time to milliseconds
            times.append(inference_time) # Appends the inference time to the list
            
            error = torch.norm(outputs - targets, dim=1) # Compute Euclidean distance error (norm of the difference between predicted and true coordinates).
            errors.extend(error.cpu().numpy()) # Moves the error values to CPU and stores them in the errors list
            # Move images, model predictions, and ground trouths to CPU and append them to there coresponding lists
            images_list.append(images.cpu()) 
            predictions.append(outputs.cpu()) 
            ground_truths.append(targets.cpu()) 
    
    errors = np.array(errors)  # Convert the error list to a numpy array for easy manipulation
    # Prints required values
    print(f"Min Distance: {errors.min():.2f}")
    print(f"Max Distance: {errors.max():.2f}")
    print(f"Mean Distance: {errors.mean():.2f}")
    print(f"Standard Deviation: {errors.std():.2f}")
    
    # Calculate accuracy within certain pixel thresholds
    thresholds = [5, 10, 20, 50] # Pixel distance thresholds
    total = len(errors) # Total number of samples
    for thresh in thresholds: # Loop through each threshold
        acc = np.sum(errors <= thresh) / total * 100 # Calculate accuracy as the percentage of samples within the threshold
        print(f"Accuracy within {thresh}px: {acc:.2f}%") # Print the accuracy for each threshold
    
    # Average inference time per image
    print(f"Average inference time per image: {np.mean(times):.2f} ms") # Calculates and prints the average inference time in milliseconds per image

    # Concatenate images, predictions, and ground trouths along the batch dimesions to create a single tensor
    images_array = torch.cat(images_list, dim=0) #
    predictions_array = torch.cat(predictions, dim=0) #
    ground_truths_array = torch.cat(ground_truths, dim=0) #
    
    sorted_indices = np.argsort(errors) # Sort errors in ascending order and get their corresponding indices
    top3_indices = sorted_indices[:3] # Get the indices of the top 3 lowest errors (best predictions)
    bottom3_indices = sorted_indices[-3:] # Get the indices of the top 3 highest errors (worst predictions)
    mean_error = errors.mean() # Calculate the mean error
    mean_indices = np.argsort(np.abs(errors - mean_error))[:3] # Get the indices of the 3 samples whose error is closest to the mean error
    
    def plot_and_save(indices, name_prefix): # Define a function to plot and save images with predictions and ground truth
        for idx in indices: # Iterate over each index to plot and save
            img = images_array[idx].permute(1, 2, 0).numpy() # Convert image tensor to numpy array for visualization (reorder dimensions)
            img = img.clip(0, 1)  # Ensure pixel values are between 0 and 1 for display
            plt.imshow(img) # Display the image using matplotlib
            pred_u, pred_v = predictions_array[idx] # Extract the predicted coordinates for the image
            gt_u, gt_v = ground_truths_array[idx] # Extract the ground truth coordinates for the image
            plt.scatter([gt_u], [gt_v], c='green', label='Ground Truth') # Plot ground truth coordinates as a green dot
            plt.scatter([pred_u], [pred_v], c='red', label='Prediction') # Plot predicted coordinates as a red dot
            plt.legend() # Show legend for ground truth and prediction
            plt.title(f'Error: {errors[idx]:.2f}px') # Set the title to display the error value
            plt.savefig(os.path.join(args.output_dir, f'{name_prefix}_{idx}.png')) # Save the image with a prefix to the output directory
            plt.close() # Close the figure after saving to free up memory
    
    # Call the function to plot and save the top 3 best, worst and closest to the mean error predictions
    plot_and_save(top3_indices, 'best')
    plot_and_save(bottom3_indices, 'worst')
    plot_and_save(mean_indices, 'mean')

if __name__ == '__main__':
    main() # Entry point of the script, calls the main function when executed