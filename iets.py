import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path

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

def get_line_pixels(x0, y0, x1, y1, width, height):
    """
    Returns the in-bounds integer pixel coordinates on a line segment.

    Uses Bresenham's line algorithm, so the result is a one-pixel-wide line
    through the 2D pixel grid.
    """
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    pixels = []
    x, y = x0, y0
    while True:
        if 0 <= x < width and 0 <= y < height:
            pixels.append((x, y))
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x += sx
        if err2 <= dx:
            err += dx
            y += sy

    return pixels

def zero_iets_except_line(grid, x0, y0, x1, y1, line_radius=0):
    """
    Returns a copy of an IET grid with values kept only near a line segment.

    Args:
        grid: 2D NumPy object array where grid[y, x] is a list of IET values.
        x0, y0: Start pixel of the line.
        x1, y1: End pixel of the line.
        line_radius: Optional integer radius around the line pixels to keep.
            Use 0 to keep only the exact one-pixel-wide line.

    Returns:
        A new grid with kept pixels copied and all other pixels set to [].
    """
    height, width = grid.shape
    line_radius = int(line_radius)
    if line_radius < 0:
        raise ValueError("line_radius must be non-negative.")

    keep_pixels = set()
    for x, y in get_line_pixels(x0, y0, x1, y1, width, height):
        for dy in range(-line_radius, line_radius + 1):
            for dx in range(-line_radius, line_radius + 1):
                nx = x + dx
                ny = y + dy
                if dx * dx + dy * dy <= line_radius * line_radius and 0 <= nx < width and 0 <= ny < height:
                    keep_pixels.add((nx, ny))

    masked_grid = np.empty_like(grid, dtype=object)
    for y in range(height):
        for x in range(width):
            masked_grid[y, x] = list(grid[y, x]) if (x, y) in keep_pixels else []

    return masked_grid

def zero_iet_file_except_line(input_filename, output_filename, x0, y0, x1, y1, line_radius=0):
    """
    Loads an IET .pkl file, keeps only IETs along a line, and saves a new .pkl.
    """
    grid = load_iet_grid(input_filename)
    masked_grid = zero_iets_except_line(grid, x0, y0, x1, y1, line_radius=line_radius)
    save_iet_grid(masked_grid, output_filename)
    return masked_grid
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --filename is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object event files.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation.")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename prefix (without extension) for the generated CSV and video. If not provided, it will be auto-generated based on rate and duration.")
    parser.add_argument("--output_filename", type=str, default=None, help="Custom output filename for the generated IET .pkl file.")
    parser.add_argument("--keep_line", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Keep IETs only along this pixel line and clear all other pixels.")
    parser.add_argument("--line_radius", type=int, default=0, help="Optional radius around --keep_line pixels to keep.")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate

    source = args.source or fm.SOURCE_NOISE
    filename = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=source,
        dataset=args.dataset,
        rate=LAMBDA_RATE,
        duration=SIM_DURATION,
        slice_name=args.slice_name,
        polarity=args.polarity,
    )
    context = fm.context_from_path(
        filename,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
        slice_name=args.slice_name,
    )
    source = context["source"]
    dataset = context["dataset"]
    slice_name = context["slice_name"]
    event_stem = Path(filename).stem
    
    event_data = pd.read_csv(filename)

    # Assuming 'event_data' is your DataFrame from previous steps
    iet_spatial_grid = create_iet_grid(event_data, SENSOR_WIDTH, SENSOR_HEIGHT)

    if args.keep_line is not None:
        x0, y0, x1, y1 = args.keep_line
        iet_spatial_grid = zero_iets_except_line(
            iet_spatial_grid,
            x0,
            y0,
            x1,
            y1,
            line_radius=args.line_radius,
        )
    
    # Save it so you don't have to re-process the CSV next time
    output_filename = (
        Path(args.output_filename)
        if args.output_filename is not None
        else fm.iet_file(
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=LAMBDA_RATE,
            duration=SIM_DURATION,
            stem=event_stem,
            slice_name=slice_name,
            polarity=args.polarity,
            create_parent=True,
        )
    )
    save_iet_grid(iet_spatial_grid, output_filename)

    # --- Example Analysis Usage ---
    # To get the 5th inter-event time at pixel x=10, y=20:
    # time_delta = iet_spatial_grid[20, 10][4]
