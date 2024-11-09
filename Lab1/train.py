# Note, many comments are added for my own learning.
import torch # Imports the core PyTorch library.
import torch.nn as nn # Imports essential building blocks like layers (Linear). nn.Linear is used for fully connected layers.
import torch.optim as optim # Imports the optimization package, which provides optimizers like Adam and SGD for updating model weights.
import torchvision.transforms as transforms # Provides tools to transform and preprocess image data (convert images to PyTorch tensors).
from torchvision.datasets import MNIST # Imports the MNIST dataset, contains 60,000 training and 10,000 test images of handwritten digits (28x28 pixels, grayscale).
from torch.utils.data import DataLoader # A utility to load the dataset in batches, which helps speed up training by processing multiple images at once.
from torchsummary import summary # Prints a summary of the model's architecture, showing each layer, output shape, and parameter count.
import matplotlib.pyplot as plt # For plotting the training loss over epochs.
from model import autoencoderMLP4Layer # Imports the autoencoderMLP4Layer class, which is the autoencoder model defined in model.py.
# Tells Python to look for a file named model.py (in the same directory or a directory specified in the Python path), and then import the autoencoderMLP4Layer class defined inside that file.
import datetime # Used to print the current time during training, mainly for logging progress.
import argparse # Allows the script to take arguments from the command line for configuration.

# Some default parameters, which can be overwritten by command line arguments.
save_file = 'MLP.8.pth.pth' # Defines the default file name where the trained model’s weights will be saved. You can overwrite this using a command-line argument.
n_epochs = 50 # Specifies the default number of training epochs (how many times the model will pass through the entire dataset).
batch_size = 2048 # The default number of training examples to process at once (a higher batch size can speed up training but requires more memory).
bottleneck_size = 8 # Sets the size of the bottleneck layer (the compressed representation of the image). This can be adjusted via command-line arguments.
plot_file = 'loss.MLP.8.png' # Specifies the file name for saving a plot of the loss function over the course of training.

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
# def train(...): This function handles the entire training process.
# n_epochs: The number of epochs to train the model.
# The optimizer used to update the model’s weights.
# model: The autoencoder model to be trained.
# loss_fn: The loss function that computes the difference between predicted and actual values.
# train_loader: The data loader that provides batches of images for training.
# scheduler: Adjusts the learning rate based on the loss.
# device: Indicates whether the model should be trained on a CPU or GPU.
# save_file: The file where model weights will be saved after training.
# plot_file: The file where the training loss plot will be saved.

    print('training ...') # Informs user that the training process has started.
    model.train() # Sets the model to training mode in PyTorch.

    losses_train = [] # Initializes an empty list to store the training loss for each epoch.
    for epoch in range(1, n_epochs+1): # Loops over the specified number of epochs (from 1 to n_epochs).

        print('epoch ', epoch) # Used to log which epoch the model is currently training on.
        loss_train = 0.0 # Initializes a variable to accumulate the training loss for the current epoch.

        # Training Loop
        for data in train_loader: # Iterates over the batches of images in the training set.
            imgs = data[0] # The images from the data batch (MNIST also contains labels, but here we only use images).
            imgs= imgs.view(imgs.shape[0], -1) # Reshapes the images to be a 2D tensor of shape (batch size, 784), where each image is flattened into a vector of size 784 (28x28 pixels).
            imgs = imgs.to(device=device) # Moves the images to the GPU if available, otherwise keeps them on the CPU.
            outputs = model(imgs) # Passes the batch of images through the autoencoder model to get reconstructed outputs.
            loss = loss_fn(outputs, imgs) # Computes the loss by comparing the reconstructed images (outputs) to the original images (imgs) using the loss function (Mean Squared Error in this case).
            optimizer.zero_grad() # Resets the gradients of all the model’s parameters to zero (required before each backward pass to avoid accumulation of gradients).
            loss.backward() # Performs backpropagation, calculating the gradients of the loss with respect to the model’s parameters.
            optimizer.step() # Updates the model’s parameters based on the gradients computed during backpropagation.
            loss_train += loss.item() # Accumulates the training loss for the current batch to track the overall loss for the epoch.

        # Learning Rate Adjustment and Loss Tracking
        scheduler.step(loss_train) # Adjusts the learning rate based on the accumulated training loss. The scheduler reduces the learning rate when the loss plateaus to fine-tune the model.
        losses_train += [loss_train/len(train_loader)] # Appends the average training loss for the current epoch to the losses_train list for plotting later. The loss is divided by the number of batches to get the average.

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader))) # Logs the current time, epoch number, and average training loss for the epoch.

        if save_file != None: # If a file name is provided, it saves the model’s state (the learned weights and biases) after each epoch using torch.save().
            torch.save(model.state_dict(), save_file) # Saves the model’s parameters (not the full model structure) to the specified file.

        if plot_file != None: # If a plot file is specified, it generates a plot of the training loss.
            plt.figure(2, figsize=(12, 7)) # Creates a new figure for the plot, with a specific size.
            plt.clf() # Clears the figure to prepare for the new plot.
            plt.plot(losses_train, label='train') # Plots the list of training losses against the number of epochs.
            plt.xlabel('epoch') # Sets the label for the x-axis (epochs).
            plt.ylabel('loss') # Sets the label for the y-axis (loss).
            plt.legend(loc=1) # Adds a legend to the plot at location 1 (upper right corner).
            print('saving ', plot_file) # Informs user that the plot is being saved.
            plt.savefig(plot_file) # Saves the plot to the specified file.

# Initializing Weights
def init_weights(m): # Initializes the weights of the model’s linear layers using the Xavier initialization, which is commonly used to prevent vanishing/exploding gradients.
    if type(m) == nn.Linear: # Checks if the module m is a linear layer.
        torch.nn.init.xavier_uniform_(m.weight) # Applies Xavier uniform initialization to the weights of the linear layer, which helps the model converge more smoothly during training.
        m.bias.data.fill_(0.01) # Sets the biases to a small positive value (0.01), initializing them slightly above zero.

def main(): # Defines the main function.
    global bottleneck_size, save_file, n_epochs, batch_size # Ensures that these variables can be modified within the main function if they are passed as command-line arguments.
    print('running main ...') # Logs a message indicating that the main function is starting.

    # Command-Line Argument Parsing
    argParser = argparse.ArgumentParser() # Initializes a parser for command-line arguments, allowing users to specify certain parameters when running the script.
    # Specifies the file name for loading or saving the model weights.
    argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    # Specifies the bottleneck size.
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    # Specifies the number of epochs.
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    # Specifies the batch size.
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    # Specifies the file name for saving the loss plot.
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    # add_argument(): Adds different arguments that can be provided.
    # -x: A short-form command-line argument that can be used to specify something.
    # metavar= 'x': Provides a label for the argument in the help text.
    # type=x: Indicates that the expected input variable type.
    # help='x': The help text that will be displayed when you run the script with --help. It describes what the argument does.

    args = argParser.parse_args() # Parses the command-line arguments into the args object.
    if args.l != None:
        save_file = args.l
    elif args.s != None:
        save_file = args.s
    if args.z != None:
        bottleneck_size = args.z
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p
    # if args.x != None:: If the -x argument is provided,
    # save_file = args.l: it sets save_file to the value of args.x.

    # Displays the current values of important parameters related to the training process.
    print('\t\tbottleneck size = ', bottleneck_size)
    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cpu' # Sets the default device to the CPU.
    if torch.cuda.is_available(): # Checks if a GPU is available.
        device = 'cuda' # If so, sets the device to cuda (the PyTorch name for GPU).
    print('\t\tusing device ', device) # Logs which device (CPU or GPU) is being used for training.

    N_input = 28 * 28  # The MNIST images are 28x28 pixels, so the input size for the model is 784 (flattened into a vector).
    N_output = N_input # Since this is an autoencoder, the output size needs to be the same as the input size (the goal is to reconstruct the original image).
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output) # Instantiates the autoencoder model with the specified input size, bottleneck size, and output size.
    model.to(device) # Moves the model to the appropriate device (CPU or GPU).
    model.apply(init_weights) # Initializes the model’s weights using the init_weights() function.
    summary(model, model.input_shape) # Prints a summary of the model’s architecture and parameter count using the torchsummary package.

    train_transform = transforms.Compose([transforms.ToTensor()]) # Applies a transformation to the training data, specifically converting images into PyTorch tensors.
    # test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) # Loads the MNIST training dataset, applying the transformation that converts images into tensors.
    # train=True: Loads the training set instead of the test set
    # download=True: If the dataset is not found locally in the given directory, it will be automatically downloaded from the internet. If false it assumes it is already downloaded.
    # transform=train_transform: Applies a transformation pipeline to the MNIST dataset before feeding it into the model.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True) # Wraps the training dataset in a DataLoader, which feeds batches of data to the model during training.
    # shuffle=True: shuffles the data in the dataset

    # Uses the Adam optimizer to update the model’s parameters during training. A learning rate of 1e-3 and a weight decay of 1e-5 are specified to prevent overfitting.
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Uses a learning rate scheduler that reduces the learning rate when the training loss plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    # Defines the loss function as Mean Squared Error (MSE), which is appropriate for image reconstruction tasks where we want to minimize the difference between the original and reconstructed images.
    loss_fn = nn.MSELoss(reduction='mean') # Changed from loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # reduction='mean': This argument specifies how the loss should be reduced (specifically for mean Averages the loss over all elements).

    # Training Call
    # Calls the train() function, passing in all the parameters necessary for training
    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
            save_file=save_file,
            plot_file=plot_file)

# This ensures that the main() function is only executed if the script is run directly (and not imported as a module).
if __name__ == '__main__':
    main()