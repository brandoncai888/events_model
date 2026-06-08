import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import file_manager as fm


def load_and_normalize_events(source, dataset, slice_name, data_root, polarity=None):
    """
    Load events from a single source/dataset/slice combination and normalize times.
    Returns a dataframe with times normalized so min(t) = 0.
    
    If source is a file path (absolute or relative), load directly from that file.
    Otherwise, construct a managed path using file_manager.
    """
    
    # Check if source is a direct file path (has .csv extension or is an existing file)
    if source.endswith('.csv') or Path(source).is_file():
        filename = Path(source)
        print(f"Loading events from: {filename}")
    else:
        # Try to construct a managed file path
        try:
            filename = fm.find_events_file(
                data_root=data_root,
                source=source,
                dataset=dataset,
                slice_name=slice_name,
                polarity=polarity,
            )
            print(f"Loading events from: {filename}")
        except (ValueError, KeyError) as e:
            # If managed path fails, try as direct path
            filename = Path(source) / dataset if dataset else Path(source)
            if not filename.exists():
                raise FileNotFoundError(
                    f"Could not find events using managed path or direct path. "
                    f"Tried source={source}, dataset={dataset}, slice={slice_name}. "
                    f"Error: {e}"
                )
            print(f"Loading events from: {filename}")
    
    df = pd.read_csv(filename)
    
    # Normalize time: shift so minimum time is 0
    min_time = df['t'].min()
    df = df.copy()
    df['t'] = df['t'] - min_time
    
    print(f"  Loaded {len(df):,} events, time range: {df['t'].min():.6f}s to {df['t'].max():.6f}s")
    
    return df


def combine_events(sources, datasets, slices, data_root, polarity=None, output_filename=None, width=None, height=None):
    """
    Combine multiple event sets by loading each, normalizing times to start at 0,
    combining them, sorting by time, and saving to output file.
    
    Args:
        sources: list of source names (e.g., ["noise", "object"])
        datasets: list of dataset names (e.g., ["1.0Hz", "45"])
        slices: list of slice names (e.g., ["1.02_1.08", None])
        data_root: root folder for managed data files
        polarity: optional polarity filter (e.g., "ON", "OFF")
        output_filename: optional output filename (default: auto-generated)
        width: optional maximum x value (keeps events with 0 <= x < width)
        height: optional maximum y value (keeps events with 0 <= y < height)
    """
    
    # Validate that we have the same number of sources, datasets, and slices
    num_sets = len(sources)
    if len(datasets) != num_sets or len(slices) != num_sets:
        raise ValueError(
            f"Must provide equal number of sources ({len(sources)}), "
            f"datasets ({len(datasets)}), and slices ({len(slices)})"
        )
    
    # Load and normalize each event set
    all_events = []
    for i, (source, dataset, slice_name) in enumerate(zip(sources, datasets, slices)):
        print(f"\n[Set {i+1}/{num_sets}]")
        df = load_and_normalize_events(source, dataset, slice_name, data_root, polarity)
        all_events.append(df)
    
    # Combine all events
    print(f"\n--- Combining {len(all_events)} event sets ---")
    combined = pd.concat(all_events, ignore_index=True)
    
    # Sort by time
    combined = combined.sort_values('t').reset_index(drop=True)
    
    print(f"Combined total: {len(combined):,} events")
    print(f"Time range: {combined['t'].min():.6f}s to {combined['t'].max():.6f}s")

    # Filter by width/height if requested
    if width is not None or height is not None:
        before_count = len(combined)
        if width is not None:
            # keep events with 0 <= x < width
            combined = combined[(combined['x'] >= 0) & (combined['x'] < float(width))]
        if height is not None:
            # keep events with 0 <= y < height
            combined = combined[(combined['y'] >= 0) & (combined['y'] < float(height))]
        combined = combined.reset_index(drop=True)
        print(f"Filtered by width/height: {before_count:,} -> {len(combined):,} events (width={width}, height={height})")
    
    # Generate output filename if not provided
    if output_filename is None:
        # Create a descriptive name from the sources and datasets
        parts = []
        for source, dataset in zip(sources, datasets):
            # Extract just the filename if source is a path
            if Path(source).exists() and Path(source).is_file():
                source_name = Path(source).stem
            else:
                source_name = source
            
            dataset_str = dataset if dataset != "-" else ""
            if dataset_str:
                parts.append(f"{source_name}_{dataset_str}")
            else:
                parts.append(source_name)
        
        combined_name = "+".join(parts)
        if polarity:
            combined_name += f"_{polarity}"
        output_filename = f"combined_events_{combined_name}.csv"
    
    # Sanitize output filename to remove any path separators
    output_filename = Path(output_filename).name
    
    # Save to CSV, preserving column order
    output_path = Path(output_filename)
    combined.to_csv(output_path, index=False, columns=['x', 'y', 't', 'p'])
    
    print(f"\n✓ Combined events saved to: {output_path}")
    print(f"  Total events: {len(combined):,}")
    print(f"  Columns: {list(combined.columns)}")
    
    return combined, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple event sets with normalized times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine two noise datasets using direct file paths
  python combine_events.py \\
    --sources data_equal_counts/poisson_noise_0.1Hz_400.0s.csv \\
            data_equal_counts/poisson_noise_10.0Hz_4.0s.csv \\
    --datasets - - --slices - -
  
  # Combine two noise managed datasets
  python combine_events.py --sources noise noise --datasets 1.0Hz 2.0Hz --slices - -
  
  # Combine three different sources with slices
  python combine_events.py \\
    --sources noise object egomotion \\
    --datasets 1.0Hz 45 1 \\
    --slices - 2.67_2.71 1.02_1.08
  
  # Combine and specify output filename
  python combine_events.py \\
    --sources noise noise \\
    --datasets 0.1Hz 10.0Hz \\
    --slices - - \\
    --output my_combined_events.csv
        """
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of source names or file paths (e.g., 'noise' 'object' or 'data_equal_counts/poisson_noise_0.1Hz_400.0s.csv')"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names or '-' if using file paths (e.g., '1.0Hz' '45' '-')"
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        required=True,
        help='List of slice names or "-" for no slice (e.g., "2.67_2.71" "-" "1.02_1.08")'
    )
    parser.add_argument(
        "--data_root",
        "--folder",
        dest="data_root",
        type=str,
        default=fm.DEFAULT_DATA_ROOT,
        help="Root folder for managed data files (default: data)"
    )
    parser.add_argument(
        "--polarity",
        choices=["ON", "OFF"],
        default=None,
        help="Optional polarity filter for object event files"
    )
    parser.add_argument(
        "--output",
        dest="output_filename",
        type=str,
        default=None,
        help="Output CSV filename (default: auto-generated)"
    )
    parser.add_argument(
        "--width",
        dest="width",
        type=float,
        default=None,
        help="Optional maximum x value; keeps events with 0 <= x < WIDTH"
    )
    parser.add_argument(
        "--height",
        dest="height",
        type=float,
        default=None,
        help="Optional maximum y value; keeps events with 0 <= y < HEIGHT"
    )
    
    args = parser.parse_args()
    
    # Convert "-" in slices to None
    slices = [None if s == "-" else s for s in args.slices]
    
    combine_events(
        sources=args.sources,
        datasets=args.datasets,
        slices=slices,
        data_root=args.data_root,
        polarity=args.polarity,
        output_filename=args.output_filename,
        width=args.width,
        height=args.height,
    )
