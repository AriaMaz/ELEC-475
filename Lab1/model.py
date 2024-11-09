# Note, many comments are added for my own learning.
import torch # Imports the core PyTorch library.
import torch.nn as nn # Imports essential building blocks like layers (Linear). nn.Linear is used for fully connected layers.
import torch.nn.functional as F # Imports functional operations like activation functions. F is used for calling functions that don’t involve learning parameters (reLU).
import torchvision.transforms as transforms # Provides tools to transform and preprocess image data (convert images to PyTorch tensors).
from torchvision.datasets import MNIST # Imports the MNIST dataset, contains 60,000 training and 10,000 test images of handwritten digits (28x28 pixels, grayscale).
from torchsummary import summary # Prints a summary of the model's architecture, showing each layer, output shape, and parameter count.

# Workaround used to avoid SSL errors when downloading datasets from a server.
import ssl # Used to handle secure connections.
ssl._create_default_https_context = ssl._create_unverified_context # Disables SSL certificate verification.

# Converts the MNIST image into a PyTorch tensor so it can be used in a model.
train_transform = transforms.Compose([transforms.ToTensor()]) #Transforms.Compose() chains multiple transformations into one.

train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
# MNIST('./data/mnist'): Loads the MNIST dataset into a specified directory (./data/mnist).
# train=True: Loads the training set (60,000 images)
# download=True: Downloads the dataset if it's not already present in the directory.
# transform=train_transform: Applies the transformation (convert images to tensors) to the dataset.

class autoencoderMLP4Layer(nn.Module): # Defines a class for the 4-layer MLP autoencoder model, which inherits from nn.Module (the base class for all PyTorch models).
    def __init__(self, N_input=28*28, N_bottleneck=8, N_output=784): # Initialization method (constructor) for the autoencoder class.
    # N_input=28*28: The input size is 28x28 pixels, flattened into a vector of 784 elements (grayscale image).
    # N_bottleneck=8: The size of the bottleneck layer (compressed latent representation) is set to 8.
    # N_output=784: The output size should be the same as the input (28x28 = 784).
        super(autoencoderMLP4Layer, self).__init__() # Calls the constructor of the parent class nn.Module, initializing necessary components for PyTorch.
        N2 = N_input // 2 # Divides the input size (784) by 2 to create an intermediate hidden layer size, N2, which equals 392. This reduces the dimensionality step by step toward the bottleneck layer.
        # nn.Linear(in_features, out_features): Defines a linear transformation, meaning each output is a weighted sum of the inputs plus a bias.
        self.fc1 = nn.Linear(N_input, N2) # Defines the first fully connected layer, fc1, that takes input of size N_input (784) and outputs size N2 (392).
        # This is where the image data is compressed into a lower-dimensional representation (bottleneck).
        self.fc2 = nn.Linear(N2, N_bottleneck) # Defines the second fully connected layer, fc2, which maps from the reduced size N2 (392) to the bottleneck size N_bottleneck (8).
        self.fc3 = nn.Linear(N_bottleneck, N2) # Defines the third fully connected layer, fc3, which reconstructs the data from the bottleneck size back to N2 (392).
        self.fc4 = nn.Linear(N2, N_output) # Defines the final fully connected layer, fc4, which reconstructs the output back to the original image size (784).
        self.type = 'MLP4' # Stores a string attribute to label this model as a 4-layer MLP autoencoder (not functionally important).
        self.input_shape = (1, N_input) #Stores the input shape of the model (batch size of 1, flattened image of size 784).

    def encode(self, X): # Defines the encoding process, which compresses the input data into the bottleneck representation.
    # X: The input tensor, which should be a flattened image of size 784.
        X = self.fc1(X) # Applies the first fully connected layer, fc1, to the input data, reducing the size from 784 to 392.
        X = F.relu(X) # Applies the ReLU (Rectified Linear Unit) activation function to the output of fc1.
        X = self.fc2(X) # Applies the second fully connected layer, fc2, further reducing the size from 392 to 8 (the bottleneck representation).
        X = F.relu(X) # Applies the ReLU activation to the output of fc2, ensuring non-linearity in the bottleneck representation.

        return X # Returns the bottleneck representation, X, which now has a size of 8.

    def decode(self, Z): # Defines the decoding process, which reconstructs the image from the bottleneck representation.
    # Z: The input tensor, which should be a bottleneck representation of size 8.
        Z = self.fc3(Z) # Applies the third fully connected layer, fc3, to the bottleneck representation, expanding the size from 8 to 392.
        Z = F.relu(Z) # Applies the ReLU (Rectified Linear Unit) activation function to the output of fc3.
        Z = self.fc4(Z) # Applies the fourth fully connected layer further transforming the data from 392 dimensions back to the original 784 dimensions.
        Z = torch.sigmoid(Z) # Applies the sigmoid activation function (maps values to a range between 0 and 1) to the output of the fc4 layer.
        # This activation is commonly used in the output layer of autoencoders, especially for images, where pixel intensities range between 0 (black) and 1 (white).

        return Z # Returns the final decoded output Z, which is the reconstructed image.

    def forward(self, X): #  Defines the forward pass of the autoencoder model.
        return self.decode(self.encode(X)) # Encodes the input X using the encode() method and then decodes it using the decode() method.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Checks if a CUDA-enabled GPU is available and, if so, sets the device to 'cuda'. If not, it defaults to using the CPU.
model = autoencoderMLP4Layer().to(device) # Instantiates the autoencoderMLP4Layer model and moves it to the selected device (GPU or CPU).

summary(model, input_size=(1, 28*28)) # Prints a detailed summary of the model architecture, including each layer's type, output shape, and number of parameters.
# Since MNIST images are 28x28 pixels, the input is a flattened vector of size 784. The 1 indicates that the model processes a batch size of 1 image at a time for the summary.