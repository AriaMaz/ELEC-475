# train.py

import torch
import torch.nn
import torch.optim as optim # Optimizers like Adam, SGD, etc.
from torch.utils.data import DataLoader
from model import SnoutNet
from dataset import PetNoseDataset
import torchvision.transforms as T
import argparse
import time # For measuring time (e.g., time per epoch)
import matplotlib.pyplot as plt
import random
from transforms import horizontal_flip, random_shift  #  Import custom data augmentation functions
import os

def apply_augmentations(image, coords, augmentations):
    """
    Apply selected data augmentations to the image and adjust coordinates accordingly.
    Args:
        image: The input image to augment.
        coords: The coordinates (u, v) to adjust for the augmentations.
        augmentations: List of augmentations to apply.
    """
    for aug in augmentations: # Iterate through the list of augmentations
        if aug == 'hflip':
            # Apply horizontal flip with probability 0.5
            if random.random() < 0.5:
                image, coords = horizontal_flip(image, coords) # Flip the image and adjust coordinates
        elif aug == 'shift':
            # Apply random shift within a maximum range
            image, coords = random_shift(image, coords, max_shift=10) # Shift the image and adjust coordinates
    return image, coords # Return the augmented image and adjusted coordinates

def main(): # Main function for training the model
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='Path to images directory')
    parser.add_argument('-t', '--train_labels', type=str, help='Path to train labels file')
    parser.add_argument('-v', '--test_labels', type=str, help='Path to test labels file')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=0, help='Weight decay (L2 penalty)')
    parser.add_argument('-a', '--augmentation', nargs='+', help='List of augmentations to apply')
    parser.add_argument('-s', '--model_save_path', type=str, default='augmentations.pth', help='Path to save the trained model')
    parser.add_argument('-p', '--loss_plot', type=str, default='loss_plot.png', help='Path to save the loss plot')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs', help='Directory to save output files')

    args = parser.parse_args() # Parse the command-line arguments

    # Ensure output directory exists
    output_dir = args.output_dir  # Use the output directory from arguments
    os.makedirs(output_dir, exist_ok=True)

    # Update paths to save model and loss plot
    model_save_path = args.model_save_path  # Save model in the directory folder
    loss_plot_path = os.path.join(output_dir, args.loss_plot)  # Save loss plot in outputs folder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU if available, otherwise use CPU
    
    transform = T.Compose([
        T.ToTensor() # Convert image to PyTorch tensor
    ])
    
    # Define the list of augmentations to apply during training
    augmentations = []
    if args.augmentation:
        augmentations = lambda img, coords: apply_augmentations(img, coords, args.augmentation) # Get augmentations from command-line arguments
    else:
        augmentations = None  # No augmentations to apply
    
    # Create the training dataset using the custom dataset class
    train_dataset = PetNoseDataset(
        images_dir=args.data_dir,  # Path to images
        labels_file=args.train_labels,  # Path to training labels file
        transform=transform,  # Apply the transformation (ToTensor)
        augmentations=augmentations  # Pass None if no augmentations
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # Create a DataLoader for the training set
    
    # Create the validation dataset (using the test dataset for validation)
    val_dataset = PetNoseDataset(
        images_dir=args.data_dir,  
        labels_file=args.test_labels,  
        transform=transform,  
        augmentations=None  # No augmentations during validation
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)  # Create a DataLoader for the validation set
    
    # Initialize the SnoutNet model
    model = SnoutNet().to(device) # Move the model to the appropriate device (CPU/GPU)
    
    # Define the optimizer (Adam) and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # Use Adam optimizer with the given learning rate and weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, # optimizer being used
                                                        'min', # learning rate will be reduced when the validation loss (or the monitored metric) stops decreasing
                                                        patience=3, # Number of epochs with no improvement after which learning rate will be reduced
                                                        factor=0.5 # specifies by how much the learning rate will be reduced once a plateau is detected (learning rate will be multiplied by this factor)
                                                     )
    
    # Early stopping parameters
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 5 # How many epochs to wait before stopping early
    epochs_no_improve = 0 # Counter for epochs without improvement
    
    # Lists to store losses and times for plotting and analysis
    epoch_losses = []
    val_losses = []
    epoch_times = []
    
    # Training loop
    for epoch in range(args.num_epochs): # Loop over the number of epochs
        model.train() # Set the model to training mode
        running_loss = 0.0 # Reset running loss for this epoch
        start_time = time.time() # Record the start time of the epoch
        
        for images, targets in train_loader: # Loop over batches of data in the training DataLoader
            images = images.to(device) # Move images to device (CPU/GPU)
            targets = targets.to(device) # Move targets to device
            
            optimizer.zero_grad() # Zero the gradients before the backward pass
            outputs = model(images) # Forward pass: get model predictions
            
            # Compute Euclidean distance loss between outputs and targets
            loss = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean() # Euclidean distance as loss
            loss.backward() # Backward pass: compute gradients
            optimizer.step() # Update model weights using the optimizer
            
            running_loss += loss.item() * images.size(0) # Accumulate the loss for this batch
        
        epoch_loss = running_loss / len(train_dataset) # Average loss for this epoch
        epoch_losses.append(epoch_loss) # Save the training loss
        
        ## Validation loop (no gradient calculations)
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0 # Initialize validation loss
        with torch.no_grad(): # Disable gradient calculation
           
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)

                loss = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean()

                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        # Step the scheduler (adjust the learning rate based on validation loss)
        scheduler.step(val_loss) # Reduce learning rate if validation loss plateau
        
        end_time = time.time()  # Record the end time of the epoch
        epoch_time = end_time - start_time # Calculate epoch duration
        epoch_times.append(epoch_time) # Store the epoch time
        
        # Print training and validation statistics for this epoch
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s')
        
        # Check for early stopping based on validation loss
        if val_loss < best_val_loss: # If validation loss improved
            best_val_loss = val_loss # Update the best validation loss
            epochs_no_improve = 0 # Reset counter for no improvement
            # Save the model checkpoint in the output directory
            torch.save(model.state_dict(), model_save_path)

        else:
            epochs_no_improve += 1 # Increment counter for no improvement
            if epochs_no_improve >= patience: # If no improvement for "patience" epochs
                print('Early stopping!') # Early stopping message
                break # Stop training
    
    # Plot the loss curve
    plt.figure() # Create a new figure for plotting
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label='Train Loss') # Plot the training loss across all epochs
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss') # Plot the validation loss across all epochs
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(loss_plot_path)  # Save the loss plot in the output directory
    avg_epoch_time = sum(epoch_times) / len(epoch_times) # Calculate the average time taken per epoch by summing the list of epoch times and dividing by the number of epochs
    print(f'Average training time per epoch: {avg_epoch_time:.2f}s') # Print the average training time per epoch, formatted to two decimal places

if __name__ == '__main__':
    main()