import numpy as np

def pad_image_to_size(original_image, padded_height, padded_width):
    # Get the dimensions of the original image
    original_height, original_width, channels = original_image.shape

    # Calculate the padding values for the top, bottom, left, and right sides
    pad_top = (padded_height - original_height) // 2
    pad_bottom = padded_height - original_height - pad_top
    pad_left = (padded_width - original_width) // 2
    pad_right = padded_width - original_width - pad_left

    # Use numpy.pad to pad the image
    padded_image = np.pad(original_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

    return padded_image
