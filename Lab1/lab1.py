# Note, many comments are added for my own learning.
# Must use numpy version 1.26.4 for the model to work.
import torch # Imports the core PyTorch library.
import torchvision.transforms as transforms # Provides tools to transform and preprocess image data (convert images to PyTorch tensors).
from torchvision.datasets import MNIST # # Imports the MNIST dataset, contains 60,000 training and 10,000 test images of handwritten digits (28x28 pixels, grayscale).
import matplotlib.pyplot as plt # For plotting the training loss over epochs.
import numpy as np # Library for numerical operations, including linear interpolation.
from model import autoencoderMLP4Layer # Imports the autoencoderMLP4Layer class, which is the autoencoder model defined in model.py.
# Tells Python to look for a file named model.py (in the same directory or a directory specified in the Python path), and then import the autoencoderMLP4Layer class defined inside that file.
import argparse # Allows the script to take arguments from the command line for configuration.

def interpolate_and_plot(model, img1, img2, device, n_steps=8):
    model.eval() # Switches / makes sure the model is in evaluation mode (turns off dropout, batch norm).

    img1 = img1.view(1, -1).to(device) # Reshapes the image to a flat vector (batch size 1, image size 784, ID tesor of size 784) and moves it to the proper device (either CPU or GPU).
    img2 = img2.view(1, -1).to(device) # Does the same for the second image.

    # Get bottleneck representations of both images
    with torch.no_grad(): # Turns off gradient calculations for efficiency and to prevent memory buildup.
        bottleneck1 = model.encode(img1)  # Encodes/compresses image bottleneck into the bottleneck (latent) representation without gradients (since gradients aren't needed during inference).
        bottleneck2 = model.encode(img2)  # Does the same for the second image.

    interpolations = [] # Initialize list to store the interpolated images.

    interpolations.append(img1.cpu().view(28, 28))  # Converts img1 back to the original shape and adds it to the start of the list.

    alphas = np.linspace(0, 1, n_steps)  # Generate n_steps evenly spaced values between 0 and 1 (exclude alpha=0 and alpha=1 for first and last images of interpolation being the actual images in the dataset).

    # Interpolate between the bottleneck tensors
    for alpha in alphas: # Loop iterates over each alpha value, where alpha represents the interpolation factor between two images.
        interpolated_bottleneck = (1 - alpha) * bottleneck1 + alpha * bottleneck2  # Calculates the linear interpolation between the two bottleneck vectors. The resulting vector is a weighted combination of the two latent representations based on the current value of alpha.
        with torch.no_grad(): # Disables gradient computation for everything inside the block.
            interpolated_img = model.decode(interpolated_bottleneck)  # The latent bottleneck vector interpolated_bottleneck is passed through the decoder of the autoencoder to obtain the interpolated image in pixel space.
            interpolations.append(interpolated_img.view(28, 28).cpu())  # Appends the reshaped interpolated imag (reshaped from a flat 1D vector (size 784) back into a 2D matrix of size 28x28 and moves the tensor from the GPU, if it was on the GPU, to the CPU).e to the interpolations list.

    # Add the original image img2 to the list
    interpolations.append(img2.cpu().view(28, 28)) # Converts img2 back to the original shape and adds it to the end of the list.

    # Plot all interpolated images including the original ones
    plt.figure(figsize=(15, 5)) # Sets up a 15 inches wide and 5 inches tall figure for plotting.
    total_images = len(interpolations) # Returns the total number of elements (images) in the interpolations list
    for i, img in enumerate(interpolations):
    # enumerate(): function that allows you to loop over a list (in this case, interpolations) and gives both the index i (the current image in the loop, starting from 0) and the value img (the current image from the interpolations list) of each element.
        plt.subplot(1, total_images, i + 1) # Creates a subplot (a smaller plot inside the main figure)
        # 1: The number of rows in the figure. In this case, 1 means that all the subplots will be arranged in a single row.
        # total_images: The number of columns (subplots) in the figure.
        # i + 1: Specifies the position of the current subplot (subplots are numbered starting from 1 index of current image, i, start at zero).
        plt.imshow(img, cmap='gray') # Displays the image img in the current subplot and sets the color map to grayscale.
        # Since the MNIST images are grayscale (each pixel is a single intensity value from 0 to 255), this ensures they are displayed correctly.
        plt.axis('off') # Turns off the axes (removes the x and y-axis labels) for a cleaner display of the image.
    plt.show() # Opens a tab that renders and displays the entire figure with all the subplots (images).

def main(): # Defines the main function.
    print('running main ...') # Log message to indicate that the main() function has started running.

    # Command-Line Argument Parsing
    argParser = argparse.ArgumentParser() # Initializes a parser for command-line arguments, allowing users to specify certain parameters when running the script.
    # Specifies the file name for loading or saving the model weights.
    argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    # Specifies the bottleneck size.
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    # add_argument(): Adds different arguments that can be provided.
    # -x: A short-form command-line argument that can be used to specify something.
    # metavar= 'x': Provides a label for the argument in the help text.
    # type=x: Indicates that the expected input variable type.
    # help='x': The help text that will be displayed when you run the script with --help. It describes what the argument does.

    args = argParser.parse_args() # Parses the command-line arguments into the args object.
    save_file = None
    if args.l != None:
        save_file = args.l
    if args.s != None:
        save_file = args.s
    if args.z != None:
        bottleneck_size = args.z
    # if args.x != None:: If the -x argument is provided,
    # save_file = args.l: it sets save_file to the value of args.x.

    device = 'cpu' # Sets the default device to the CPU.
    if torch.cuda.is_available(): # Checks if a GPU is available.
        device = 'cuda' # If so, sets the device to cuda (the PyTorch name for GPU).
    print('\t\tusing device ', device) # Logs which device (CPU or GPU) is being used for training.

    train_transform = transforms.Compose([transforms.ToTensor()]) # Applies a transformation to the training data, specifically converting images into PyTorch tensors.
    test_transform = train_transform # Use the same transformation for the test dataset.

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) # Loads the MNIST training dataset, applying the transformation that converts images into tensors.
    # train=True: Loads the training set instead of the test set
    # download=True: If the dataset is not found locally in the given directory, it will be automatically downloaded from the internet. If false it assumes it is already downloaded.
    # transform=train_transform: Applies a transformation pipeline to the MNIST dataset before feeding it into the model.

    # test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)

    N_input = 28 * 28 # The MNIST images are 28x28 pixels, so the input size for the model is 784 (flattened into a vector).
    bottleneck_size = 8 # Sets bottleneck to be 8.
    N_output = N_input # Since this is an autoencoder, the output size needs to be the same as the input size (the goal is to reconstruct the original image).
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output) # Instantiates the autoencoder model with the specified input size, bottleneck size, and output size.
    model.load_state_dict(torch.load(save_file)) # Load the saved weights into the model.
    model.to(device) # Moves the model to the appropriate device (CPU or GPU).
    model.eval() # Puts the model in evaluation mode. This disables operations like dropout and ensures that batch normalization works in inference mode.

    # idx = 0

    # Loop allows the user to enter an image index (from the MNIST dataset) for processing.
    while True:
        idx = input("Enter index (or type nothing ad click enter to move to interpolation) > ")

        # If the user presses Enter without typing anything, the loop breaks and the script moves to the interpolation section.
        if idx.lower() == '':
            break

        idx = int(idx) # Convert the user input (string) to an integer index.
        if 0 <= idx <= train_set.data.size()[0]: # Ensure the index is valid.
            print('label = ', train_set.targets[idx].item()) # Print the label (digit) of the selected image.
            img = train_set.data[idx] # Get the image from the dataset based on the user-provided index.
            # print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32) # Convert the image to `float32` type (uses 32 bits (4 bytes) in memory to represent a real number).
            # print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = img / 255.0 # Normalize the pixel values to the range [0, 1].
            # print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            mode = input("Enter 'r' for reconstruction, 'd' for denoising > ") # Ask user if they want to reconstruct or denoise.

            if mode == 'd': # If denoising is selected.
                noise = (torch.rand(img.shape) - 0.5) * 1.25 # Create noise.
                input_img = img + noise # Add random noise to the original image.
                input_img = torch.clamp(input_img, 0., 1.)  # Ensure pixel values are still between 0 and 1
            else: # If reconstruction is selected, no noise added
                input_img = img # If reconstruction is selected, use the original image.

            input_img = input_img.to(device) # Move the image to the correct device (CPU or GPU).
            input_img = input_img.view(1, -1) # Flatten the image to a 1D tensor.

            with torch.no_grad(): # Disable gradient calculations (not needed for inference).
                output = model(input_img) # Pass the image (either with or without noise) through the autoencoder.

            # print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor) # Reshapes img into a 2D tensor with shape (1, 784), where 784 is the result of 28 * 28 (the flattened version of a 28x28 image).
            # print('break 9 : ', img.shape, img.dtype)
            output = output.view(28, 28).type(torch.FloatTensor) # Reshapes the output tensor (which was initially a flattened vector of size 784) back into a 2D image of size 28x28 and converts the reshaped tensor to torch.FloatTensor to ensure it’s in floating-point format.
            # print('break 10 : ', output.shape, output.dtype)
            # print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))

            # Ensure the image is properly detached and converted to numpy for display
            img = img.view(28, 28).cpu().numpy()  # Reshapes the img tensor back to its original 28x28 image dimensions, Ensures the tensor is moved back to the CPU (if it was on a GPU), and Converts the PyTorch tensor to a NumPy array.
            input_img = input_img.view(28, 28).cpu().numpy()  # Does the same for input_img
            output = output.view(28, 28).cpu().numpy()  # Does the same for input_img

            if mode == 'd': # If denoising is selected.
                f = plt.figure()
                f.add_subplot(1, 3, 1)
                plt.imshow(img, cmap='gray')
                plt.title('Original Image')

                f.add_subplot(1, 3, 2)
                plt.imshow(input_img, cmap='gray')
                plt.title('Noisy Image')

                f.add_subplot(1, 3, 3)
                plt.imshow(output, cmap='gray')
                plt.title('Denoised Output')
                plt.show()

            else: # If reconstruction is selected
                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.title('Original Image')

                f.add_subplot(1, 2, 2)
                plt.imshow(output, cmap='gray')
                plt.title('Reconstructed Output')
                plt.show()
    # f = plt.figure(): Creates a new figure object for plotting and stores it in f
    # f.add_subplot(x, y, z): Adds a subplot (a smaller plot inside the larger figure) to the figure f. Subplots are used to display multiple images or graphs side-by-side or in a grid.
        # x: Specifies the number of rows in the grid.
        # y: Specifies the number of columns in the grid.
        # z: Specifies the position of this subplot in the grid. The subplots are numbered from left to right, top to bottom, starting at 1.
    # plt.imshow(img, cmap='gray'): Displays the image img in the current subplot (created in the previous step). The image will be displayed in grayscale.
    # plt.title('x'): Adds a title above the current subplot, labeling the image as the "x".

    # Image Interpolation
    idx1 = int(input("Enter index for first image (for interpolation) > ")) # Get first image index from user.
    idx2 = int(input("Enter index for second image (for interpolation) > ")) # Get second image index from user.

    if 0 <= idx1 <= train_set.data.size()[0] and 0 <= idx2 <= train_set.data.size()[0]: # Ensure the index's are valid.
        print(f'Label for first image: {train_set.targets[idx1].item()}') # Print the label for first image.
        print(f'Label for second image: {train_set.targets[idx2].item()}') # Do the same for the second image.

        img1 = train_set.data[idx1].float() / 255.0  # Convert the image to float type and normalize the pixel values to the range [0, 1].
        img2 = train_set.data[idx2].float() / 255.0  # Do the same for the second image.

        interpolate_and_plot(model, img1, img2, device, n_steps=8) # Call the function to interpolate between the two images and plot the result.

# This ensures that the main() function is only executed if the script is run directly (and not imported as a module).
if __name__ == '__main__':
    main()