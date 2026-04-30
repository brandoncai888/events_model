import pandas as pd
import numpy as np
import pickle

def create_iet_grid(df, width, height):
    """
    Groups events by pixel and stores their inter-event times in a 2D grid.
    
    Returns:
        A NumPy array of shape (height, width) where each element is a Python list.
    """
    print("Step 1: Calculating inter-event times...")
    # Sorting ensures dt is calculated chronologically per pixel
    df = df.sort_values(by=['x', 'y', 't'])
    df['dt'] = df.groupby(['x', 'y'])['t'].diff()
    
    # Drop the first event of each pixel (which is always NaN)
    df_clean = df.dropna(subset=['dt'])

    print("Step 2: Initializing 2D object grid...")
    # Create a grid of empty lists
    grid = np.empty((height, width), dtype=object)
    for y in range(height):
        for x in range(width):
            grid[y, x] = []

    print("Step 3: Populating buckets...")
    # Group by coordinates and extract the dt series
    grouped = df_clean.groupby(['y', 'x'])['dt']
    
    for (y, x), values in grouped:
        # Converting to list for easier manipulation later
        grid[y, x] = values.tolist()

    return grid

def save_iet_grid(grid, filename="iet_grid.pkl"):
    """Saves the 2D array of lists to a binary file."""
    with open(filename, 'wb') as f:
        pickle.dump(grid, f)
    print(f"Grid successfully saved to {filename}")

def load_iet_grid(filename="iet_grid.pkl"):
    """Loads the 2D array of lists from a binary file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

if __name__ == "__main__":
    # Example dimensions (e.g., DAVIS346)
    W, H = 346, 260
    
    # Assuming 'event_data' is your DataFrame from previous steps
    iet_spatial_grid = create_iet_grid(event_data, W, H)
    
    # Save it so you don't have to re-process the CSV next time
    save_iet_grid(iet_spatial_grid, "my_event_analysis.pkl")

    # --- Example Analysis Usage ---
    # To get the 5th inter-event time at pixel x=10, y=20:
    # time_delta = iet_spatial_grid[20, 10][4]