import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dataset import SingleTransformerDataset

def calculate_kappa_values_with_exclusion(dataset, exclude_values=[-2.0, -1.0], buffer_factor=1.1):
    """
    Calculate kappa values for x and y coordinates from decoder_input,
    excluding specific token values.
    
    Parameters:
    dataset: SingleTransformerDataset instance
    exclude_values: list of values to exclude from kappa calculation
    buffer_factor: multiplier to add some buffer space
    
    Returns:
    kappa_x, kappa_y: range parameters for x and y coordinates
    """
    # Initialize min/max trackers
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    # Create a dataloader to iterate through the dataset
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    for batch in dataloader:
        decoder_input = batch['decoder_input']
        
        # Create masks to exclude special token values
        x_mask = torch.ones_like(decoder_input[..., 0], dtype=torch.bool)
        y_mask = torch.ones_like(decoder_input[..., 1], dtype=torch.bool)
        
        for exclude_val in exclude_values:
            x_mask = x_mask & (decoder_input[..., 0] != exclude_val)
            y_mask = y_mask & (decoder_input[..., 1] != exclude_val)
        
        # Get valid x and y coordinates
        valid_x = decoder_input[..., 0][x_mask]
        valid_y = decoder_input[..., 1][y_mask]
        
        if len(valid_x) > 0:
            x_min = min(x_min, valid_x.min().item())
            x_max = max(x_max, valid_x.max().item())
        
        if len(valid_y) > 0:
            y_min = min(y_min, valid_y.min().item())
            y_max = max(y_max, valid_y.max().item())
    
    # Calculate kappa values (maximum absolute value from origin)
    kappa_x = max(abs(x_min), abs(x_max)) * buffer_factor
    kappa_y = max(abs(y_min), abs(y_max)) * buffer_factor
    
    return kappa_x, kappa_y

def analyze_decoder_input_with_exclusion(dataset, exclude_values=[-2.0, -1.0], num_samples=5):
    """
    Helper function to understand the structure of decoder_input with exclusions.
    """
    print("Analyzing decoder_input structure (excluding special tokens)...")
    print(f"Excluding values: {exclude_values}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        decoder_input = sample['decoder_input']
        
        # Create masks
        x_mask = torch.ones_like(decoder_input[..., 0], dtype=torch.bool)
        y_mask = torch.ones_like(decoder_input[..., 1], dtype=torch.bool)
        
        for exclude_val in exclude_values:
            x_mask = x_mask & (decoder_input[..., 0] != exclude_val)
            y_mask = y_mask & (decoder_input[..., 1] != exclude_val)
        
        valid_x = decoder_input[..., 0][x_mask]
        valid_y = decoder_input[..., 1][y_mask]
        
        print(f"Sample {i}: shape={decoder_input.shape}")
        print(f"  Total points: {len(decoder_input)}")
        print(f"  Valid x points: {len(valid_x)}")
        print(f"  Valid y points: {len(valid_y)}")
        if len(valid_x) > 0:
            print(f"  X range: [{valid_x.min():.3f}, {valid_x.max():.3f}]")
        if len(valid_y) > 0:
            print(f"  Y range: [{valid_y.min():.3f}, {valid_y.max():.3f}]")
        print()

# Alternative faster approach (if dataset fits in memory)
def calculate_kappa_values_fast_with_exclusion(data_dir, exclude_values=[-2.0, -1.0], buffer_factor=1.1):
    """
    Faster approach if the entire dataset fits in memory.
    """
    decoder_input = np.load(os.path.join(data_dir, "decoder_input.npy"))
    
    # Create masks to exclude special values
    x_mask = np.ones_like(decoder_input[..., 0], dtype=bool)
    y_mask = np.ones_like(decoder_input[..., 1], dtype=bool)
    
    for exclude_val in exclude_values:
        x_mask = x_mask & (decoder_input[..., 0] != exclude_val)
        y_mask = y_mask & (decoder_input[..., 1] != exclude_val)
    
    # Get valid coordinates
    valid_x = decoder_input[..., 0][x_mask]
    valid_y = decoder_input[..., 1][y_mask]
    
    print(f"Found {len(valid_x)} valid x coordinates and {len(valid_y)} valid y coordinates")
    
    # Calculate kappa with buffer
    kappa_x = max(np.abs(valid_x.min()), np.abs(valid_x.max())) * buffer_factor
    kappa_y = max(np.abs(valid_y.min()), np.abs(valid_y.max())) * buffer_factor
    
    return kappa_x, kappa_y

# Usage example
if __name__ == "__main__":
    data_dir = "/home/anurizada/Documents/nobari_10_transformer"
    
    # Create dataset
    dataset = SingleTransformerDataset(data_dir)
    
    # Analyze the structure with exclusions
    analyze_decoder_input_with_exclusion(dataset)
    
    # Calculate kappa values excluding special tokens
    kappa_x, kappa_y = calculate_kappa_values_with_exclusion(dataset)
    
    print(f"Kappa values calculated (excluding special tokens):")
    print(f"Kappa_x (for x coordinates): {kappa_x:.4f}")
    print(f"Kappa_y (for y coordinates): {kappa_y:.4f}")
    print(f"X coordinate range: [-{kappa_x:.4f}, {kappa_x:.4f}]")
    print(f"Y coordinate range: [-{kappa_y:.4f}, {kappa_y:.4f}]")

# Kappa values calculated (excluding special tokens):
# Kappa_x (for x coordinates): 1.6501
# Kappa_y (for y coordinates): 1.6500
# X coordinate range: [-1.6501, 1.6501]
# Y coordinate range: [-1.6500, 1.6500]