from html import parser

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from iets import load_iet_grid
import argparse
from pathlib import Path

import file_manager as fm


def save_binned_values_csv(save_path, bin_edges, data_values, expected_values):
    """
    Saves plotted binned values to CSV.

    bin, left, and right are stored as log10-transformed positions.
    """
    csv_path = save_path + '.csv'
    log_left = np.log10(bin_edges[:-1])
    log_right = np.log10(bin_edges[1:])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin', 'left', 'right', 'data', 'expected'])
        for (i, left, right, data, expected) in zip(
            range(len(log_left)), log_left, log_right, data_values, expected_values
        ):
            writer.writerow([i, left, right, data, expected])
    print(f"CSV saved successfully to {csv_path}")

def calculate_pixel_frequencies(grid):
    """
    Calculates the average firing frequency (Hz) for each pixel.
    Frequency is defined as 1 / (average Inter-Event Time).
    
    Args:
        grid: A 2D NumPy array of lists containing IETs (shape H x W).
        
    Returns:
        A 2D NumPy array of the same shape containing float frequencies.
    """
    print("Calculating average frequencies per pixel...")
    H, W = grid.shape
    freq_map = np.zeros((H, W), dtype=float)
    
    for y in range(H):
        for x in range(W):
            iets = grid[y, x]
            if len(iets) > 0:
                mean_dt = np.mean(iets)
                if mean_dt > 0:
                    freq_map[y, x] = 1.0 / mean_dt
                    
    return freq_map

def plot_frequency_pdf(
    freq_map,
    bins=100,
    color='coral',
    min=None,
    max=None,
    expected_rate=None,
    expected_std_dev=None,
    save_path=None,
    suppress_show=False
):
    """
    Graphs the Probability Density Function (PDF) of pixel frequencies.
    Uses a log scale for the X-axis (Frequency).

    Args:
        freq_map: 2D NumPy array of pixel frequencies.
        bins: Number of histogram bins.
        color: Plot color.
        min: Minimum frequency for bin range.
        max: Maximum frequency for bin range.
        expected_rate: Optional rate used for Gaussian curve overlay.
        expected_std_dev: Optional standard deviation for Gaussian curve overlay.
        save_path: Optional path to save the resulting plot image.
    """
    print("Generating PDF of average pixel frequencies...")
    
    # Flatten the 2D map into a 1D array and filter out dead pixels (0 Hz)
    # We must filter out 0s because log(0) is undefined.
    freqs = freq_map.flatten()
    valid_freqs = freqs[freqs > 0]
    
    if len(valid_freqs) == 0:
        print("No valid frequency data to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Create logarithmically spaced bins between the min and max frequencies
    if min is not None and max is not None:
        min_val, max_val = min, max
    else:
        min_val, max_val = valid_freqs.min(), valid_freqs.max()
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=bins)
    
    # density=True converts counts to probability density
    data_density, _, _ = plt.hist(valid_freqs, bins=bin_edges, density=True, log=False, color=color, edgecolor='black', alpha=0.8)
    expected_density = np.full_like(data_density, np.nan, dtype=float)

    if expected_rate is not None and expected_std_dev is not None:
        rate = expected_rate
        std_dev = expected_std_dev

        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        z_left = (bin_edges[:-1] - rate) / (std_dev * np.sqrt(2.0))
        z_right = (bin_edges[1:] - rate) / (std_dev * np.sqrt(2.0))
        cdf_left = 0.5 * (1.0 + np.array([math.erf(z) for z in z_left]))
        cdf_right = 0.5 * (1.0 + np.array([math.erf(z) for z in z_right]))
        expected_bin_areas = cdf_right - cdf_left
        expected_density = expected_bin_areas / bin_widths

        label = f'Expected Gaussian (mu={rate}, sigma={std_dev:.3g})'

        plt.plot(bin_centers, expected_density, color='orange', lw=2, linestyle='--', label=label)
        plt.legend()

    # Set X-axis to logarithmic scale
    plt.xscale('log')

    plt.title("PDF of Average Pixel Frequencies")
    plt.xlabel("Average Frequency (Hz) [Log Scale]")
    plt.ylabel("Probability Density")
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        save_binned_values_csv(save_path, bin_edges, data_density, expected_density)
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        print(f"Image saved successfully to {save_path+'.png'}")
    if not suppress_show:
        plt.show()

def plot_overall_iet_histogram(
    grid,
    bins=100,
    color='teal',
    min=None,
    max=None,
    expected_rate=None,
    expected_total=None,
    expected_std_dev=None,
    expected_log_mean=None,
    expected_log_std_dev=None,
    expected_shape=None,
    save_path=None,
    suppress_show=False
):
    """
    Graphs the overall histogram of ALL inter-event times across the entire sensor.
    Uses a log scale for the X-axis.

    If expected_shape is provided, overlays expected bin counts for that distribution.

    Args:
        grid: 2D array of IET lists.
        bins: Number of histogram bins.
        color: Plot color.
        min: Minimum IET for bin range.
        max: Maximum IET for bin range.
        expected_rate: Optional rate/mean used for expected curve overlay.
        expected_total: Optional total count used for expected curve overlay.
        expected_std_dev: Optional standard deviation used for Gaussian expected counts.
        expected_log_mean: Optional mean of ln(IET) for log-normal expected counts.
        expected_log_std_dev: Optional standard deviation of ln(IET) for log-normal expected counts.
        expected_shape: Optional expected shape: 'exponential', 'gaussian', or 'lognormal'.
        save_path: Optional path to save the resulting plot image.
    """
    print("Generating overall histogram of Inter-Event Times...")
    
    # Extract all IETs from the 2D grid of lists into a single flat list
    # List comprehension is the fastest way to unpack this structure in Python
    all_iets = [dt for row in grid for dt_list in row for dt in dt_list]
    
    # Convert to NumPy array and filter out zero or negative values
    all_iets_arr = np.array(all_iets, dtype=float)
    valid_iets = all_iets_arr[np.isfinite(all_iets_arr) & (all_iets_arr > 0)]
    
    if len(valid_iets) == 0:
        print("No valid IET data to plot.")
        return
    plt.figure(figsize=(10, 6))
    
    # Create bins for the Time (X) axis
    if min is not None and max is not None:
        min_val, max_val = min, max
    else:
        min_val, max_val = valid_iets.min(), valid_iets.max()
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=bins)

    data_counts, _, _ = plt.hist(valid_iets, bins=bin_edges, log=False, color=color, edgecolor='none', alpha=0.8)
    expected_counts = np.full_like(data_counts, np.nan, dtype=float)

    if expected_shape is not None and expected_total is None:
        expected_total = data_counts.sum()

    if expected_shape == 'gaussian':
        rate = expected_rate
        std_dev = expected_std_dev
        if rate is None or std_dev is None:
            print("Expected Gaussian curve skipped: expected_rate and expected_std_dev are required.")
        elif std_dev <= 0:
            print("Expected Gaussian curve skipped: expected_std_dev must be positive.")
        else:
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
            z_left = (bin_edges[:-1] - rate) / (std_dev * np.sqrt(2.0))
            z_right = (bin_edges[1:] - rate) / (std_dev * np.sqrt(2.0))
            cdf_left = 0.5 * (1.0 + np.array([math.erf(z) for z in z_left]))
            cdf_right = 0.5 * (1.0 + np.array([math.erf(z) for z in z_right]))
            expected_counts = expected_total * (cdf_right - cdf_left)

            label = f'Expected Gaussian (mu={rate}, sigma={std_dev:.3g}, total={expected_total})'

            plt.plot(bin_centers, expected_counts, color='orange', lw=2, linestyle='--', label=label)
            plt.legend()
            
    elif expected_shape == 'exponential':
        rate = expected_rate
        if rate is None:
            print("Expected exponential curve skipped: expected_rate is required.")
        elif rate <= 0:
            print("Expected exponential curve skipped: expected_rate must be positive.")
        else:
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
            expected_counts = expected_total * (
                np.exp(-rate * bin_edges[:-1]) - np.exp(-rate * bin_edges[1:])
            )

            label = f'Expected exponential (lambda={rate}, total={expected_total:g})'

            plt.plot(bin_centers, expected_counts, color='orange', lw=2, linestyle='--', label=label)
            plt.legend()
    elif expected_shape == 'lognormal':
        plotted_iets = valid_iets[(valid_iets >= min_val) & (valid_iets <= max_val)]
        if expected_log_mean is None or expected_log_std_dev is None:
            if len(plotted_iets) < 2:
                print("Expected log-normal curve skipped: at least two plotted IETs are required to fit parameters.")
            else:
                log_iets = np.log(plotted_iets)
                expected_log_mean = np.mean(log_iets)
                expected_log_std_dev = np.std(log_iets, ddof=1)

        if expected_log_mean is not None and expected_log_std_dev is not None:
            if expected_log_std_dev <= 0:
                print("Expected log-normal curve skipped: expected_log_std_dev must be positive.")
            else:
                bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
                z_left = (np.log(bin_edges[:-1]) - expected_log_mean) / (expected_log_std_dev * np.sqrt(2.0))
                z_right = (np.log(bin_edges[1:]) - expected_log_mean) / (expected_log_std_dev * np.sqrt(2.0))
                cdf_left = 0.5 * (1.0 + np.array([math.erf(z) for z in z_left]))
                cdf_right = 0.5 * (1.0 + np.array([math.erf(z) for z in z_right]))
                expected_counts = expected_total * (cdf_right - cdf_left)

                label = (
                    f'Expected log-normal (mu={expected_log_mean:.3g}, '
                    f'sigma={expected_log_std_dev:.3g}, total={expected_total:g})'
                )

                plt.plot(bin_centers, expected_counts, color='orange', lw=2, linestyle='--', label=label)
                plt.legend()

    # Set X-axis to logarithmic scale
    plt.xscale('log')
    
    plt.title("Overall Distribution of Inter-Event Times (All Pixels)")
    plt.xlabel("Inter-Event Time (Seconds) [Log Scale]")
    plt.ylabel("Event Count")
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        save_binned_values_csv(save_path, bin_edges, data_counts, expected_counts)
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        print(f"Image saved successfully to {save_path+'.png'}")
    if not suppress_show:
        plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--rate", type=float, default=None, help="Poisson event rate per pixel in Hz.")
    parser.add_argument(
        "--expected_shape",
        choices=["exponential", "gaussian", "lognormal", "none"],
        default="exponential",
        help="Expected curve to overlay on the IET histogram.",
    )
    parser.add_argument("--std_dev", type=float, default=None, help="Standard deviation for Gaussian curve overlay.")
    parser.add_argument("--log_mean", type=float, default=None, help="Mean of ln(IET) for log-normal curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--log_std_dev", type=float, default=None, help="Standard deviation of ln(IET) for log-normal curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--expected_total", "--exp_total", dest="expected_total", type=float, default=None, help="Total count used to scale the expected overlay. Defaults to the plotted histogram count.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --filename is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object IET files.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation.")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename prefix (without extension) for the generated CSV and video. If not provided, it will be auto-generated based on rate and duration.")
    parser.add_argument("--no_expected", action="store_true", help="Suppress expected curve overlays on the plots.")
    parser.add_argument("--min_iet", type=float, default=1e-4, help="Minimum IET to consider for plotting (default: 1e-4 seconds).")
    parser.add_argument("--max_iet", type=float, default=10.0, help="Maximum IET to consider for plotting (default: 100 seconds).")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate
    MIN_RES = 1e-5
    source = args.source or fm.SOURCE_NOISE
    filename = fm.find_iet_file(
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
    plot_stem = Path(filename).stem
    if plot_stem.endswith("_iet"):
        plot_stem = plot_stem[:-4]
    
    # Assuming 'iet_spatial_grid' is the 2D array of lists created in the previous step
    iet_spatial_grid = load_iet_grid(filename)
    
    '''
    # Graph the Frequency PDF
    freq_map = calculate_pixel_frequencies(iet_spatial_grid)
    # 40 bins per decade
    # for 346*260 = 89960 pixels spanning ~1 decade
    plot_frequency_pdf( 
        freq_map, bins=241, color='red', 
        min=1, max=10000.0,
        expected_rate=LAMBDA_RATE if not args.no_expected else None,
        expected_std_dev=np.sqrt(LAMBDA_RATE / (SIM_DURATION)) if not args.no_expected else None,
        save_path=str(fm.picture_base(
            "frequency_pdf",
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=LAMBDA_RATE,
            duration=SIM_DURATION,
            stem=plot_stem,
            slice_name=slice_name,
            create_parent=True,
        )),
        suppress_show=args.no_show
        )
    '''

    # Graph the overall IET distribution
    args.min_iet = max(args.min_iet, MIN_RES)
    num_bins = int(math.log10(args.max_iet / args.min_iet) * 100 + 1.5)

    expected_shape = None if args.no_expected or args.expected_shape == 'none' else args.expected_shape

    plot_overall_iet_histogram(
        iet_spatial_grid, bins=num_bins, color='blue', 
        min=args.min_iet, max=args.max_iet, 
        expected_rate=args.rate,
        expected_total=args.expected_total,
        expected_std_dev=args.std_dev,
        expected_log_mean=args.log_mean,
        expected_log_std_dev=args.log_std_dev,
        expected_shape=expected_shape,
        save_path=str(fm.picture_base(
            "iet_hist",
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=LAMBDA_RATE,
            duration=SIM_DURATION,
            stem=plot_stem,
            slice_name=slice_name,
            create_parent=True,
        )),
        suppress_show=args.no_show
        )
