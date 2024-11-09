# transforms.py

import random # Imports the random module to generate random numbers.
import torchvision.transforms.functional as F # Imports functional transformations from torchvision, specifically for image manipulations.

def horizontal_flip(image, coords):
    """
    Custom data augmentation method to horizontally flip the image and adjust coordinates.
    """
    image = F.hflip(image) # Horizontally flips the image using torchvision's hflip function
    width, _ = image.size # Gets the width of the image, the underscore is used to ignore the height value
    u, v = coords # Unpacks the (u, v) coordinates
    u = width - u # Adjusts the 'u' coordinate by subtracting it from the width (flips the coordinate horizontally)
    return image, (u, v) # Returns the flipped image and the new coordinates

def random_shift(image, coords, max_shift=10): # Shift 10 pixels
    """
    Custom data augmentation method to randomly shift the image and adjust coordinates.
    """
    # Generates a random horizontal and vertical shift value between -max_shift and +max_shift
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    u, v = coords # Unpacks the (u, v) coordinates
    image = F.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0) # Applies an affine transformation to shift the image by (dx, dy), no rotation, scaling, or shearing is applied
    # Adjust coordinates by adding the horizontal and vertical shift values
    u_new = u + dx
    v_new = v + dy

    # Ensure coordinates are within image bounds
    width, height = image.size # Gets the width and height of the image

    # Ensures the new 'u, v' coordinate is within the image's width and height (from 0 to width/height-1)
    u_new = max(0, min(u_new, width - 1)) # Ensures the new 'u' coordinate is within the image's width (from 0 to width-1)
    v_new = max(0, min(v_new, height - 1))  # Ensures the new 'u' coordinate is within the image's width (from 0 to width-1).
    return image, (u_new, v_new) # Returns the shifted image and the updated (u, v) coordinates