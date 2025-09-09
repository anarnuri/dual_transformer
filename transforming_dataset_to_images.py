import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = "/home/anurizada/Documents/nobari_10_joints"
# Load your curves data
curves = np.load(os.path.join(data_dir, "curves.npy"), mmap_mode='r')

# Digitization function
def digitize_seq(nums, minlim, maxlim, bin_size=64):
    bins = np.linspace(minlim, maxlim, bin_size-1)
    nums_indices = np.digitize(nums, bins)
    return nums_indices

# Image creation function
def get_image_from_point_cloud(points, xylim, im_size, inverted=True, label=None):
    mat = np.zeros((im_size, im_size, 1), dtype=np.uint8)
    x = digitize_seq(points[:,0], -xylim, xylim, im_size)
    if inverted:
        y = digitize_seq(points[:,1]*-1, -xylim, xylim, im_size)
        mat[y, x, 0] = 1
    else:
        y = digitize_seq(points[:,1], -xylim, xylim, im_size)
        mat[x, y, 0] = 1
    return mat

# Process all curves and save as NPY
def process_all_curves_to_npy(curves, output_path, xylim=1.8, im_size=64, inverted=True):
    """
    Process all curves into binary images and save as a single NPY file.
    """
    print(f"Curves array shape: {curves.shape}")
    print(f"Total curves to process: {len(curves)}")
    
    # Pre-allocate array for efficiency (if we know the exact number)
    # images = np.zeros((len(curves), im_size, im_size, 1), dtype=np.uint8)
    
    # Or use list and convert to array later (more memory efficient for unknown sizes)
    images_list = []
    
    # Process all curves with progress bar
    for i in tqdm(range(len(curves)), desc="Processing curves"):
        curve = curves[i]
        
        # Make sure the curve has the right shape (N, 2)
        if curve.ndim == 2 and curve.shape[1] == 2:
            # Create image from point cloud
            image = get_image_from_point_cloud(curve, xylim, im_size, inverted)
            images_list.append(image)
        else:
            print(f"Warning: Curve {i} has unexpected shape {curve.shape}")
            # Add empty image or skip - here we'll add a blank image
            blank_image = np.zeros((im_size, im_size, 1), dtype=np.uint8)
            images_list.append(blank_image)
    
    # Convert list to numpy array
    images_array = np.array(images_list)
    print(f"Generated images array shape: {images_array.shape}")
    
    # Save as NPY file
    np.save(output_path, images_array)
    print(f"Saved {len(images_list)} images to {output_path}")
    
    return images_array

# Process all curves and save to file
output_file_path = os.path.join(data_dir, "curve_images.npy")
all_images = process_all_curves_to_npy(curves, output_file_path, inverted=True)

# Optional: Verify the saved file
def verify_saved_file(file_path):
    """Verify the saved NPY file can be loaded correctly."""
    try:
        loaded_images = np.load(file_path)
        print(f"Verified saved file: {loaded_images.shape}")
        print(f"Data type: {loaded_images.dtype}")
        print(f"Value range: [{loaded_images.min()}, {loaded_images.max()}]")
        return loaded_images
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Verify the saved file
verify_saved_file(output_file_path)