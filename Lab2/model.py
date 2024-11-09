# model.py

import torch # Imports the core PyTorch library
import torch.nn as nn # Imports essential building blocks like layers (Linear). nn.Linear is used for fully connected layers
import torch.nn.functional as F # Imports functional operations like activation functions. F is used for calling functions that don’t involve learning parameters (reLU)

class SnoutNet(nn.Module): # Define the neural network class SnoutNet
    def __init__(self): # Constructor to define the layers of the network
        super(SnoutNet, self).__init__() # Calls the parent class (nn.Module) to initialize its properties
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d( # Applies a 2D convolution
            in_channels=3, # The input has 3 channels (RGB image)
            out_channels=64, # This layer will produce 64 feature maps
            kernel_size=3, # The convolutional kernel has a size of 3x3
            stride=1, # The kernel moves 1 pixel at a time
            padding=1 # 1-pixel padding keeps the output size the same as the input size
        )
        # Max Pooling after Conv1
        self.pool1 = nn.MaxPool2d(  # Creates a max pooling layer
        # Max pooling is a downsampling operation that reduces the size of feature maps by taking the maximum value in a specified window
            kernel_size=4,  # 4x4 window, meaning the layer will look at 4x4 blocks and pick the maximum value from each block.
            stride=4, 
            padding=1
        )
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        # Max Pooling after Conv2
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        # Max Pooling after Conv3
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)
        
        # Fully Connected Layers
        # After Conv3 and Pooling, the feature map is 4x4x256 = 4096
        self.fc1 = nn.Linear(4 * 4 * 256, 1024) # The input is flattened to 4096 input and 1024 output
        self.fc2 = nn.Linear(1024, 1024) # Another fully connected layer with 1024 input and output units
        self.fc3 = nn.Linear(1024, 2)  # Output is 2 coordinates (u, v)
        
        # Initialize weights for the layers
        self._initialize_weights()  # Calls the custom weight initialization function defined below

    def forward(self, x): # Defines the forward pass through the network layers.
        # Convolutional Layer 1 with ReLU and Max Pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Convolutional Layer 2 with ReLU and Max Pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Convolutional Layer 3 with ReLU and Max Pooling
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten the feature maps into a single vector
        x = x.view(-1, 4 * 4 * 256) # Reshapes the output tensor into a flat vector (batch_size, 4096).
        # -1 in this context is a placeholder that allows PyTorch to infer the correct size for that dimension based on the remaining dimensions.
        
        # Passes the flattened vector through Fully Connected Layers and applies ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output Layer
        x = self.fc3(x) # Final layer that outputs 2 values (u, v coordinates).
        
        return x # Returns the output of the network.
    
    def _initialize_weights(self): # Custom weight initialization for all layers
        for m in self.modules(): # Loops through each module (layer) in the network
            if isinstance(m, nn.Conv2d): # If the layer is a convolutional layer ...
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Initializes the weights using Kaiming normal initialization for ReLU activation
                if m.bias is not None: # If the layer has a bias ...
                    nn.init.constant_(m.bias, 0.0) # Initializes the bias to 0
            elif isinstance(m, nn.Linear): # If the layer is a fully connected (linear) layer ...
                nn.init.xavier_uniform_(m.weight) # Initializes the weights using Xavier uniform initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01) # Initializes the bias to 0.01.