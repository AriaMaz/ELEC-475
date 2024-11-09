# model_test.py

import torch
from model import SnoutNet # Importing the SnoutNet model from model.py

def main():
    # Create a dummy input tensor of shape (1, 3, 227, 227)
    dummy_input = torch.randn(1, 3, 227, 227) # torch.randn generates random values from a normal distribution
    # Initialize the model
    model = SnoutNet() # Creates an instance of the SnoutNet model
    # Forward pass
    output = model(dummy_input) # Pass the dummy input through the model to get the output
    # Print the output shape
    print("Output shape:", output.shape)  # Displays the shape of the model's output
    print("Output:", output) # This will print the actual values predicted by the model for the dummy input tensor

if __name__ == '__main__':
    main()

# run: python model_test.py

# Expected Output:
# Output shape: torch.Size([1, 2])
# Output: tensor([[value1, value2]], grad_fn=<AddmmBackward0>)
# The values will be floating point numbers corresponding to the predicted coordinates.