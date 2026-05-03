from html import parser

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from .inter_event_time import load_iet_grid
import argparse
import sys


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
        if std_dev <= 0:
            print("Expected Gaussian curve skipped: expected_std_dev must be positive.")
        else:
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
    save_path=None,
    suppress_show=False
):
    """
    Graphs the overall histogram of ALL inter-event times across the entire sensor.
    Uses a log scale for the X-axis.

    If expected_rate is provided, overlays the expected exponential bin counts.
    Set fit_rate=True to estimate lambda from the plotted IET data instead.

    Args:
        grid: 2D array of IET lists.
        bins: Number of histogram bins.
        color: Plot color.
        min: Minimum IET for bin range.
        max: Maximum IET for bin range.
        expected_rate: Optional rate used for expected curve overlay.
        expected_total: Optional total count used for expected curve overlay.
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
    
    # Create logarithmically spaced bins for the Time (X) axis
    if min is not None and max is not None:
        min_val, max_val = min, max
    else:
        min_val, max_val = valid_iets.min(), valid_iets.max()
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=bins)
    
    data_counts, _, _ = plt.hist(valid_iets, bins=bin_edges, log=False, color=color, edgecolor='black', alpha=0.8)
    expected_counts = np.full_like(data_counts, np.nan, dtype=float)
    
    if expected_rate is not None and expected_total is not None:
        rate = expected_rate
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        expected_counts = expected_total * (
            np.exp(-rate * bin_edges[:-1]) - np.exp(-rate * bin_edges[1:])
        )

        label = f'Expected exponential (lambda={rate}, total={expected_total})'

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
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--folder", type=str, default="data", help="Base folder to save results (default: data).")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation.")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate
    SUFFIX = f"{LAMBDA_RATE}Hz_{SIM_DURATION}s"

    MIN_TIME = 1e-6 # minimum IET due to typical numerical precision limits
    
    # Assuming 'iet_spatial_grid' is the 2D array of lists created in the previous step
    iet_spatial_grid = load_iet_grid(f"{args.folder}/poisson_noise_{SUFFIX}_iet.pkl")
    
    # Graph the Frequency PDF
    freq_map = calculate_pixel_frequencies(iet_spatial_grid)
    # 100 bins per decade seems reasonable 
    # for 346*260 = 89960 pixels spanning ~1 decade
    plot_frequency_pdf( 
        freq_map, bins=401, color='red', 
        min=LAMBDA_RATE/100.0, max=LAMBDA_RATE*100.0,
        expected_rate=LAMBDA_RATE,
        expected_std_dev=np.sqrt(LAMBDA_RATE / (SIM_DURATION)),
        save_path=f"{args.folder}/poisson_noise_{SUFFIX}_frequency_pdf",
        suppress_show=args.no_show
        )
    
    # Graph the overall IET distribution
    num_bins = int(math.log10(SIM_DURATION / MIN_TIME) * 100 + 1.5) 
    # 100 bins per decade definitely reasonable 
    # for 39*346*260 = 3596000 IETs spanning ~2 decades
    plot_overall_iet_histogram(
        iet_spatial_grid, bins=num_bins, color='blue', 
        min=MIN_TIME, max=SIM_DURATION, 
        expected_rate=LAMBDA_RATE, 
        expected_total=SENSOR_WIDTH * SENSOR_HEIGHT * (SIM_DURATION * LAMBDA_RATE - 1),
        save_path=f"{args.folder}/poisson_noise_{SUFFIX}_iet_hist",
        suppress_show=args.no_show
        )
