import numpy as np
import matplotlib.pyplot as plt

from inter_event_time import load_iet_grid

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

def plot_frequency_pdf(freq_map, bins=100, color='coral', min=None, max=None):
    """
    Graphs the Probability Density Function (PDF) of pixel frequencies.
    Uses a log scale for both the X-axis (Frequency) and Y-axis (Density).
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
    # log=True automatically sets the Y-axis to a logarithmic scale
    plt.hist(valid_freqs, bins=bin_edges, density=True, log=False, color=color, edgecolor='black', alpha=0.8)
    
    # Set X-axis to logarithmic scale
    plt.xscale('log')
    
    plt.title("PDF of Average Pixel Frequencies")
    plt.xlabel("Average Frequency (Hz) [Log Scale]")
    plt.ylabel("Probability Density [Log Scale]")
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_overall_iet_histogram(
    grid,
    bins=100,
    color='teal',
    min=None,
    max=None,
    expected_rate=None,
    fit_rate=False,
):
    """
    Graphs the overall histogram of ALL inter-event times across the entire sensor.
    Uses a log scale for the X-axis.

    If expected_rate is provided, overlays the expected exponential bin counts.
    Set fit_rate=True to estimate lambda from the plotted IET data instead.
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
    
    plt.hist(valid_iets, bins=bin_edges, log=False, color=color, edgecolor='black', alpha=0.8)
    
    if expected_rate is not None or fit_rate:
        rate = 1.0 / valid_iets.mean() if fit_rate else expected_rate
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        expected_counts = len(valid_iets) * (
            np.exp(-rate * bin_edges[:-1]) - np.exp(-rate * bin_edges[1:])
        )

        if fit_rate:
            label = f'Fitted exponential (lambda={rate:.3f})'
        else:
            label = f'Expected exponential (lambda={rate})'

        plt.plot(bin_centers, expected_counts, color='orange', lw=2, linestyle='--', label=label)
        plt.legend()


    # Set X-axis to logarithmic scale
    plt.xscale('log')
    
    plt.title("Overall Distribution of Inter-Event Times (All Pixels)")
    plt.xlabel("Inter-Event Time (Seconds) [Log Scale]")
    plt.ylabel("Event Count [Log Scale]")
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Assuming 'iet_spatial_grid' is the 2D array of lists created in the previous step
    iet_spatial_grid = load_iet_grid("poisson_noise_5.0.pkl")
    
    # 1. Calculate the frequency map
    freq_map = calculate_pixel_frequencies(iet_spatial_grid)
    
    # 2. Graph the Frequency PDF
    plot_frequency_pdf(freq_map, bins=100, color='red', min=None, max=None)
    
    # 3. Graph the overall IET distribution
    plot_overall_iet_histogram(iet_spatial_grid, bins=150, color='blue', min=None, max=None, expected_rate=5.0)
