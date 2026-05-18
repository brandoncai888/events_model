import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import file_manager as fm


LINE_PART_RE = re.compile(r"^line(\d+)$")
POLARITIES = {"ON", "OFF"}


def load_events_csv(csv_path):
    """Load events from a CSV file."""
    print(f"Loading events from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    required_cols = {'x', 'y', 't'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_cols)}")
    
    print(f"Loaded {len(df):,} events")
    return df


def filter_by_polarity(df, polarity):
    """Filter events by polarity if specified."""
    if polarity is None:
        return df
    
    if 'p' not in df.columns:
        raise ValueError("Polarity column 'p' not found in CSV")
    
    # Convert polarity string to appropriate value
    if polarity == 'ON':
        polarity_val = 1
    elif polarity == 'OFF':
        polarity_val = 0
    else:
        raise ValueError(f"Polarity must be 'ON' or 'OFF'; got {polarity}")
    
    filtered = df[df['p'] == polarity_val].copy()
    print(f"Filtered to {len(filtered):,} events with polarity={polarity}")
    return filtered


def create_event_count_map(df, width, height):
    """
    Create a 2D histogram of event counts.
    
    Args:
        df: DataFrame with 'x' and 'y' columns
        width: Image width
        height: Image height
    
    Returns:
        A 2D numpy array with event counts at each pixel
    """
    print("Creating event count map...")
    
    # Ensure events are within bounds
    in_bounds = (df['x'] >= 0) & (df['x'] < width) & (df['y'] >= 0) & (df['y'] < height)
    if not in_bounds.all():
        dropped = len(df) - int(in_bounds.sum())
        print(f"Warning: dropping {dropped:,}/{len(df):,} events outside the {width}x{height} frame")
        df = df[in_bounds].copy()
    
    # Create 2D histogram using numpy
    count_map = np.zeros((height, width), dtype=np.uint32)
    
    # Round coordinates to nearest integer
    x_coords = np.round(df['x']).astype(int)
    y_coords = np.round(df['y']).astype(int)
    
    # Clip to ensure within bounds
    x_coords = np.clip(x_coords, 0, width - 1)
    y_coords = np.clip(y_coords, 0, height - 1)
    
    # Increment counts
    np.add.at(count_map, (y_coords, x_coords), 1)
    
    max_count = count_map.max()
    print(f"Event count map created. Max events at a pixel: {max_count}")
    
    return count_map, max_count


def split_count_stem(stem, polarity=None, line=None):
    """Return a canonical base stem plus detected polarity and line parts."""
    detected_polarity = polarity
    detected_line = line
    base_parts = []

    for part in str(stem).split("_"):
        if part in POLARITIES:
            detected_polarity = detected_polarity or part
            continue

        line_match = LINE_PART_RE.match(part)
        if line_match:
            detected_line = detected_line or int(line_match.group(1))
            continue

        base_parts.append(part)

    return "_".join(base_parts), detected_polarity, detected_line


def next_power_of_two(value):
    """Return the smallest power of two greater than or equal to value."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def powers_of_two_ticks(max_count):
    """Return colorbar tick labels at zero and exponential count steps."""
    ticks = [0]
    if max_count < 1:
        return ticks

    max_tick = next_power_of_two(max_count)
    value = 1
    while value <= max_tick:
        ticks.append(value)
        value *= 2
    return ticks


def log2_count_scale(count_map):
    """Map counts so 0, 1, 2, 4, 8... are equally spaced in color."""
    scaled = np.zeros(count_map.shape, dtype=float)
    positive = count_map > 0
    scaled[positive] = np.log2(count_map[positive]) + 1.0
    return scaled


def save_count_visualization(count_map, output_path, width, height, max_count, title="Event Count Heatmap", show=True):
    """
    Save a visualization of the event count map with a colorbar.
    
    Args:
        count_map: 2D numpy array of event counts
        output_path: Path to save the image
        width: Original image width
        height: Original image height
        max_count: Maximum count value for scaling
        title: Title for the plot
        show: Whether to display the image after saving it
    """
    print("Creating visualization...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")

    colorbar_ticks = powers_of_two_ticks(max_count)
    colorbar_positions = [0 if tick == 0 else np.log2(tick) + 1.0 for tick in colorbar_ticks]
    scaled_counts = log2_count_scale(count_map)
    scale_max = max(colorbar_positions)

    # Color is linear over this transformed scale: 0, 1, 2, 4, 8... are equally spaced.
    im = ax.imshow(
        scaled_counts,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=scale_max,
    )
    
    # Add colorbar legend
    cbar = plt.colorbar(im, ax=ax, label="Event Count", pad=0.02)
    cbar.set_ticks(colorbar_positions)
    cbar.set_ticklabels([str(tick) for tick in colorbar_ticks])
    cbar.ax.tick_params(colors='white')
    cbar.ax.yaxis.label.set_color('white')
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, color="white", fontsize=14, pad=20)
    ax.set_xlabel("X coordinate", color="white")
    ax.set_ylabel("Y coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("black")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a heatmap showing event counts per pixel."
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --dataset is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object event files.")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename for the events CSV file.")
    
    parser.add_argument("--output_filename", type=str, default=None, help="Custom output filename for the generated IET .pkl file.")
    parser.add_argument("--keep_line", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Keep IETs only along this pixel line and clear all other pixels.")
    parser.add_argument("--line_radius", type=int, default=0, help="Optional radius around --keep_line pixels to keep.")
    parser.add_argument("--line", type=int, default=None, help="Use line-filtered events file with the specified line number (e.g., --line 1 for line1.csv).")
    
    parser.add_argument("--output", type=str, default=None, help="Custom output path for the heatmap image.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the image.")
    
    args = parser.parse_args()
    
    source_resolved = args.source or fm.SOURCE_NOISE
    
    filename = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=source_resolved,
        dataset=args.dataset,
        rate=args.rate,
        duration=args.duration,
        slice_name=args.slice_name,
        polarity=args.polarity,
        line=args.line,
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
    
    csv_path = Path(filename)
    if not csv_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {csv_path}")
    
    # Load and process events
    df = load_events_csv(csv_path)
    df = filter_by_polarity(df, args.polarity)
    
    if df.empty:
        print("ERROR: No events to process after filtering")
        sys.exit(1)
    
    # Create count map
    count_map, max_count = create_event_count_map(df, args.width, args.height)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        base_stem, output_polarity, output_line = split_count_stem(
            Path(filename).stem,
            polarity=args.polarity,
            line=args.line,
        )
        output_path = fm.picture_file(
            "counts",
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            slice_name=slice_name,
            stem=base_stem,
            polarity=output_polarity,
            line=output_line,
            create_parent=True,
        )
    
    # Create title
    title = f"Event Count Heatmap | {Path(filename).name}"
    if args.polarity:
        title += f" | Polarity: {args.polarity}"
    
    # Save visualization
    save_count_visualization(
        count_map,
        output_path,
        args.width,
        args.height,
        max_count,
        title=title,
        show=not args.no_show,
    )
