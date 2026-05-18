import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import file_manager as fm

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
    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--dataset", "--set", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz'.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation.")
    parser.add_argument("--filename", type=str, default=None, help="Optional explicit input CSV path.")
    parser.add_argument("--output_filename", type=str, default=None, help="Optional explicit output IET .pkl path.")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate

    filename = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=fm.SOURCE_NOISE,
        dataset=args.dataset,
        rate=LAMBDA_RATE,
        duration=SIM_DURATION,
    )
    
    event_data = pd.read_csv(filename)

    # Assuming 'event_data' is your DataFrame from previous steps
    iet_spatial_grid = create_iet_grid(event_data, SENSOR_WIDTH, SENSOR_HEIGHT)
    
    # Save it so you don't have to re-process the CSV next time
    output_filename = (
        Path(args.output_filename)
        if args.output_filename is not None
        else fm.iet_file(
            data_root=args.data_root,
            source=fm.SOURCE_NOISE,
            dataset=args.dataset,
            rate=LAMBDA_RATE,
            duration=SIM_DURATION,
            stem=Path(filename).stem,
            create_parent=True,
        )
    )
    save_iet_grid(iet_spatial_grid, output_filename)

    # --- Example Analysis Usage ---
    # To get the 5th inter-event time at pixel x=10, y=20:
    # time_delta = iet_spatial_grid[20, 10][4]
