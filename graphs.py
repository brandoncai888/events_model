from html import parser

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from iets import load_iet_grid
import argparse
from pathlib import Path

import file_manager as fm


def save_binned_values_csv(save_path, bin_edges, data_values, expected_values, xscale="log"):
    """
    Saves plotted binned values to CSV.

    For log-scale plots, left and right are stored as log10-transformed positions.
    For linear-scale plots, left and right are stored as raw positions.
    """
    csv_path = save_path + '.csv'
    if xscale == "log":
        left_values = np.log10(bin_edges[:-1])
        right_values = np.log10(bin_edges[1:])
    elif xscale == "linear":
        left_values = bin_edges[:-1]
        right_values = bin_edges[1:]
    else:
        raise ValueError("xscale must be either 'log' or 'linear'.")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin', 'left', 'right', 'data', 'expected'])
        for (i, left, right, data, expected) in zip(
            range(len(left_values)), left_values, right_values, data_values, expected_values
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

    plt.figure(figsize=(8, 5))
    
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

def _normal_cdf(x, mean, std_dev):
    z = (x - mean) / (std_dev * np.sqrt(2.0))
    return 0.5 * (1.0 + np.array([math.erf(value) for value in z]))


def _normal_pdf(x, mean, std_dev):
    return np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2.0 * np.pi))


def _lognormal_cdf(x, log_mean, log_std_dev):
    z = (np.log(x) - log_mean) / (log_std_dev * np.sqrt(2.0))
    return 0.5 * (1.0 + np.array([math.erf(value) for value in z]))


def _lognormal_pdf(x, log_mean, log_std_dev):
    return (
        np.exp(-0.5 * ((np.log(x) - log_mean) / log_std_dev) ** 2)
        / (x * log_std_dev * np.sqrt(2.0 * np.pi))
    )


def _standard_normal_cdf(x):
    return 0.5 * (1.0 + np.array([math.erf(value / np.sqrt(2.0)) for value in x]))


def _log_standard_normal_cdf(x):
    x = np.asarray(x, dtype=float)
    log_cdf = np.empty_like(x, dtype=float)
    regular = x > -10.0
    if np.any(regular):
        log_cdf[regular] = np.log(_standard_normal_cdf(x[regular]))
    if np.any(~regular):
        tail = x[~regular]
        log_cdf[~regular] = -0.5 * tail ** 2 - np.log(-tail) - 0.5 * np.log(2.0 * np.pi)
    return log_cdf


def _inverse_gaussian_cdf(x, mean, shape):
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x, dtype=float)
    positive = x > 0
    if np.any(positive):
        xp = x[positive]
        sqrt_shape_over_x = np.sqrt(shape / xp)
        z1 = sqrt_shape_over_x * (xp / mean - 1.0)
        z2 = -sqrt_shape_over_x * (xp / mean + 1.0)
        log_second_term = 2.0 * shape / mean + _log_standard_normal_cdf(z2)
        second_term = np.exp(np.minimum(log_second_term, np.log(np.finfo(float).max)))
        cdf[positive] = _standard_normal_cdf(z1) + second_term
    return np.clip(cdf, 0.0, 1.0)


def _inverse_gaussian_pdf(x, mean, shape):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x, dtype=float)
    positive = x > 0
    if np.any(positive):
        xp = x[positive]
        pdf[positive] = (
            np.sqrt(shape / (2.0 * np.pi * xp ** 3))
            * np.exp(-shape * (xp - mean) ** 2 / (2.0 * mean ** 2 * xp))
        )
    return pdf


def _exponential_cdf(x, rate):
    return 1.0 - np.exp(-rate * x)


def _exponential_pdf(x, rate):
    return rate * np.exp(-rate * x)


def _adaptive_linear_bounds(values, padding_fraction=0.02):
    min_val = np.min(values)
    max_val = np.max(values)
    span = max_val - min_val
    if span <= 0:
        span = max(abs(min_val), 1.0)
    padding = span * padding_fraction
    lower = max(np.nextafter(0.0, 1.0), min_val - padding)
    upper = max_val + padding
    if upper <= lower:
        upper = lower + span
    return lower, upper


def _resolve_expected_iet_params(
    plotted_iets,
    expected_shape,
    expected_rate,
    expected_std_dev,
    expected_log_mean,
    expected_log_std_dev,
    expected_invgauss_mean,
    expected_invgauss_shape,
):
    if expected_shape == "gaussian":
        expected_shape = "normal"
    elif expected_shape == "log-normal":
        expected_shape = "lognormal"
    elif expected_shape in ("inverse-gaussian", "invgauss"):
        expected_shape = "inverse_gaussian"

    if expected_shape == "normal":
        if expected_rate is None:
            expected_rate = np.mean(plotted_iets)
        if expected_std_dev is None:
            if len(plotted_iets) < 2:
                print("Expected normal curve skipped: at least two plotted IETs are required to fit standard deviation.")
            else:
                expected_std_dev = np.std(plotted_iets, ddof=1)
        if expected_std_dev is not None and expected_std_dev <= 0:
            print("Expected normal curve skipped: expected_std_dev must be positive.")
            expected_std_dev = None
    elif expected_shape == "exponential":
        if expected_rate is None:
            mean_iet = np.mean(plotted_iets)
            if mean_iet > 0:
                expected_rate = 1.0 / mean_iet
        if expected_rate is not None and expected_rate <= 0:
            print("Expected exponential curve skipped: expected_rate must be positive.")
            expected_rate = None
    elif expected_shape == "lognormal":
        if expected_log_mean is None or expected_log_std_dev is None:
            if len(plotted_iets) < 2:
                print("Expected log-normal curve skipped: at least two plotted IETs are required to fit parameters.")
            else:
                log_iets = np.log(plotted_iets)
                expected_log_mean = np.mean(log_iets)
                expected_log_std_dev = np.std(log_iets, ddof=1)
        if expected_log_std_dev is not None and expected_log_std_dev <= 0:
            print("Expected log-normal curve skipped: expected_log_std_dev must be positive.")
            expected_log_std_dev = None
    elif expected_shape == "inverse_gaussian":
        if expected_invgauss_mean is None:
            expected_invgauss_mean = np.mean(plotted_iets)
        if expected_invgauss_shape is None:
            if len(plotted_iets) < 2:
                print("Expected inverse Gaussian curve skipped: at least two plotted IETs are required to fit parameters.")
            elif expected_invgauss_mean > 0:
                denominator = np.sum(
                    (plotted_iets - expected_invgauss_mean) ** 2
                    / (expected_invgauss_mean ** 2 * plotted_iets)
                )
                if denominator > 0:
                    expected_invgauss_shape = len(plotted_iets) / denominator
        if expected_invgauss_mean is not None and expected_invgauss_mean <= 0:
            print("Expected inverse Gaussian curve skipped: expected_invgauss_mean must be positive.")
            expected_invgauss_mean = None
        if expected_invgauss_shape is not None and expected_invgauss_shape <= 0:
            print("Expected inverse Gaussian curve skipped: expected_invgauss_shape must be positive.")
            expected_invgauss_shape = None

    return (
        expected_shape,
        expected_rate,
        expected_std_dev,
        expected_log_mean,
        expected_log_std_dev,
        expected_invgauss_mean,
        expected_invgauss_shape,
    )


def _expected_iet_values(
    bin_edges,
    bin_centers,
    expected_shape,
    distribution,
    expected_rate,
    expected_total,
    expected_std_dev,
    expected_log_mean,
    expected_log_std_dev,
    expected_invgauss_mean,
    expected_invgauss_shape,
):
    if expected_shape == "normal":
        if expected_rate is None or expected_std_dev is None:
            return None, None
        label = f'Expected normal (mu={expected_rate:.3g}, sigma={expected_std_dev:.3g})'
        if distribution == "pdf":
            return _normal_pdf(bin_centers, expected_rate, expected_std_dev), label
        values = expected_total * (
            _normal_cdf(bin_edges[1:], expected_rate, expected_std_dev)
            - _normal_cdf(bin_edges[:-1], expected_rate, expected_std_dev)
        )
        return values, f'{label}, total={expected_total:g}'

    if expected_shape == "exponential":
        if expected_rate is None:
            return None, None
        label = f'Expected exponential (lambda={expected_rate:.3g})'
        if distribution == "pdf":
            return _exponential_pdf(bin_centers, expected_rate), label
        values = expected_total * (
            _exponential_cdf(bin_edges[1:], expected_rate)
            - _exponential_cdf(bin_edges[:-1], expected_rate)
        )
        return values, f'{label}, total={expected_total:g}'

    if expected_shape == "lognormal":
        if expected_log_mean is None or expected_log_std_dev is None:
            return None, None
        label = f'Expected log-normal (mu={expected_log_mean:.3g}, sigma={expected_log_std_dev:.3g})'
        if distribution == "pdf":
            return _lognormal_pdf(bin_centers, expected_log_mean, expected_log_std_dev), label
        values = expected_total * (
            _lognormal_cdf(bin_edges[1:], expected_log_mean, expected_log_std_dev)
            - _lognormal_cdf(bin_edges[:-1], expected_log_mean, expected_log_std_dev)
        )
        return values, f'{label}, total={expected_total:g}'

    if expected_shape == "inverse_gaussian":
        if expected_invgauss_mean is None or expected_invgauss_shape is None:
            return None, None
        label = f'Expected inverse Gaussian (mu={expected_invgauss_mean:.3g}, lambda={expected_invgauss_shape:.3g})'
        if distribution == "pdf":
            return _inverse_gaussian_pdf(bin_centers, expected_invgauss_mean, expected_invgauss_shape), label
        values = expected_total * (
            _inverse_gaussian_cdf(bin_edges[1:], expected_invgauss_mean, expected_invgauss_shape)
            - _inverse_gaussian_cdf(bin_edges[:-1], expected_invgauss_mean, expected_invgauss_shape)
        )
        return values, f'{label}, total={expected_total:g}'

    return None, None


def plot_overall_iet_distribution(
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
    expected_invgauss_mean=None,
    expected_invgauss_shape=None,
    expected_shape=None,
    distribution="pdf",
    xscale="log",
    save_path=None,
    suppress_show=False
):
    """
    Graphs the overall distribution of ALL inter-event times across the entire sensor.
    Uses either a log or linear scale for the X-axis.

    If expected_shape is provided, overlays the expected distribution.

    Args:
        grid: 2D array of IET lists.
        bins: Number of histogram bins.
        color: Plot color.
        min: Minimum IET for bin range.
        max: Maximum IET for bin range.
        expected_rate: Optional rate/mean used for expected curve overlay.
        expected_total: Optional total count used for expected curve overlay.
        expected_std_dev: Optional standard deviation used for normal expected curve.
        expected_log_mean: Optional mean of ln(IET) for log-normal expected curve.
        expected_log_std_dev: Optional standard deviation of ln(IET) for log-normal expected curve.
        expected_invgauss_mean: Optional mean for inverse Gaussian expected curve.
        expected_invgauss_shape: Optional shape lambda for inverse Gaussian expected curve.
        expected_shape: Optional expected shape: 'exponential', 'normal', 'gaussian', 'lognormal', 'log-normal', 'inverse_gaussian', 'inverse-gaussian', or 'invgauss'.
        distribution: Either 'pdf' or 'histogram'. PDF is normalized to probability density.
        xscale: Either 'log' or 'linear'. Linear mode uses adaptive bounds if min/max are not provided.
        save_path: Optional path to save the resulting plot image.
    """
    if distribution not in ("pdf", "histogram"):
        raise ValueError("distribution must be either 'pdf' or 'histogram'.")
    if xscale not in ("log", "linear"):
        raise ValueError("xscale must be either 'log' or 'linear'.")

    print(f"Generating overall IET {distribution} on a {xscale} x-scale...")
    
    # Extract all IETs from the 2D grid of lists into a single flat list
    # List comprehension is the fastest way to unpack this structure in Python
    all_iets = [dt for row in grid for dt_list in row for dt in dt_list]
    
    # Convert to NumPy array and filter out zero or negative values
    all_iets_arr = np.array(all_iets, dtype=float)
    valid_iets = all_iets_arr[np.isfinite(all_iets_arr) & (all_iets_arr > 0)]
    
    if len(valid_iets) == 0:
        print("No valid IET data to plot.")
        return
    plt.figure(figsize=(8, 5))
    
    # Create bins for the Time (X) axis.
    if max is not None:
        min_val, max_val = min, max
    elif xscale == "linear":
        min_val, max_val = _adaptive_linear_bounds(valid_iets)
        min_val = 0
    else:
        min_val, max_val = valid_iets.min(), valid_iets.max()
    if xscale == "log":
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=bins)
    else:
        bin_edges = np.linspace(min_val, max_val, num=bins)

    plotted_iets = valid_iets[(valid_iets >= min_val) & (valid_iets <= max_val)]
    if len(plotted_iets) == 0:
        print("No valid IET data within the requested plot range.")
        return

    is_pdf = distribution == "pdf"
    data_label = f"Data (events={len(plotted_iets):,})"
    data_values, _, _ = plt.hist(
        valid_iets,
        bins=bin_edges,
        density=is_pdf,
        log=False,
        color=color,
        edgecolor='none',
        alpha=0.8,
        label=data_label,
    )
    expected_values = np.full_like(data_values, np.nan, dtype=float)
    if xscale == "log":
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if expected_shape is not None:
        expected_total = data_values.sum() if expected_total is None else expected_total
        (
            expected_shape,
            expected_rate,
            expected_std_dev,
            expected_log_mean,
            expected_log_std_dev,
            expected_invgauss_mean,
            expected_invgauss_shape,
        ) = _resolve_expected_iet_params(
            plotted_iets,
            expected_shape,
            expected_rate,
            expected_std_dev,
            expected_log_mean,
            expected_log_std_dev,
            expected_invgauss_mean,
            expected_invgauss_shape,
        )
        expected, label = _expected_iet_values(
            bin_edges,
            bin_centers,
            expected_shape,
            distribution,
            expected_rate,
            expected_total,
            expected_std_dev,
            expected_log_mean,
            expected_log_std_dev,
            expected_invgauss_mean,
            expected_invgauss_shape,
        )
        if expected is not None:
            expected_values = expected
            plt.plot(bin_centers, expected_values, color='orange', lw=2, linestyle='--', label=label)

    plt.xscale(xscale)
    
    plot_name = "PDF" if is_pdf else "Histogram"
    plt.title(f"Overall IET {plot_name} (All Pixels)")
    scale_label = "Log Scale" if xscale == "log" else "Linear Scale"
    plt.xlabel(f"Inter-Event Time (Seconds) [{scale_label}]")
    plt.ylabel("Probability Density" if is_pdf else "Event Count")
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        save_binned_values_csv(save_path, bin_edges, data_values, expected_values, xscale=xscale)
        plt.savefig(save_path+'.png', dpi=300, bbox_inches='tight')
        print(f"Image saved successfully to {save_path+'.png'}")
    if not suppress_show:
        plt.show()


def plot_overall_iet_histogram(*args, **kwargs):
    kwargs["distribution"] = "histogram"
    return plot_overall_iet_distribution(*args, **kwargs)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate IET distribution graphs.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--rate", type=float, default=None, help="Poisson event rate per pixel in Hz.")
    parser.add_argument(
        "--expected_shape",
        choices=["exponential", "normal", "gaussian", "lognormal", "log-normal", "inverse_gaussian", "inverse-gaussian", "invgauss", "none"],
        default="exponential",
        help="Expected curve to overlay on the IET distribution. 'gaussian' is accepted as an alias for 'normal'.",
    )
    parser.add_argument(
        "--distribution",
        choices=["pdf", "histogram"],
        default="pdf",
        help="IET distribution plot type. Defaults to PDF.",
    )
    parser.add_argument("--std_dev", type=float, default=None, help="Standard deviation for normal curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--log_mean", type=float, default=None, help="Mean of ln(IET) for log-normal curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--log_std_dev", type=float, default=None, help="Standard deviation of ln(IET) for log-normal curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--invgauss_mean", type=float, default=None, help="Mean mu for inverse Gaussian curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--invgauss_shape", type=float, default=None, help="Shape lambda for inverse Gaussian curve overlay. If omitted, fit from plotted data.")
    parser.add_argument("--expected_total", "--exp_total", dest="expected_total", type=float, default=None, help="Total count used to scale the expected overlay. Defaults to the plotted histogram count.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --filename is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object IET files.")
    parser.add_argument("--line", type=int, default=None, help="Optional keep_line label to graph, for example --line 1.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation.")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename prefix (without extension) for the generated CSV and video. If not provided, it will be auto-generated based on rate and duration.")
    parser.add_argument("--no_expected", action="store_true", help="Suppress expected curve overlays on the plots.")
    parser.add_argument("--linear", action="store_true", help="Plot IET distribution on a linear x-scale with adaptive bounds and save with a _lin suffix.")
    parser.add_argument("--min_iet", type=float, default=None, help="Minimum IET to consider for plotting (default: 1e-4 seconds).")
    parser.add_argument("--max_iet", type=float, default=None, help="Maximum IET to consider for plotting (default: 100 seconds).")
    parser.add_argument("--bins_per_decade", type=int, default=100, help="Number of bins per decade for log plots; linear plots use this as the number of bins (default: 100).")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate
    MIN_RES = 1e-5
    source = args.source or fm.SOURCE_NOISE
    iet_stem = None
    if args.line is not None:
        if args.line < 1:
            raise ValueError("--line must be 1 or greater.")
        base_iet = fm.iet_file(
            data_root=args.data_root,
            source=source,
            dataset=args.dataset,
            rate=LAMBDA_RATE,
            duration=SIM_DURATION,
            slice_name=args.slice_name,
            polarity=args.polarity,
        )
        base_stem = base_iet.stem
        if base_stem.endswith("_iet"):
            base_stem = base_stem[:-4]
        iet_stem = f"{base_stem}_line{args.line}"
    filename = fm.find_iet_file(
        filename=args.filename,
        data_root=args.data_root,
        source=source,
        dataset=args.dataset,
        rate=LAMBDA_RATE,
        duration=SIM_DURATION,
        stem=iet_stem,
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

    # Graph the overall IET distribution. PDF is the default; the legacy count
    # histogram remains available with --distribution histogram.
    if args.linear:
        min_iet = 0
        max_iet = args.max_iet
        num_bins = args.bins_per_decade + 1
        xscale = "linear"
    else:
        if args.min_iet is None and args.max_iet is None:
            min_iet = 1e-4
            max_iet = 1.0
        else:
            args.min_iet = 1e-4 if args.min_iet is None else args.min_iet
            args.max_iet = 1.0 if args.max_iet is None else args.max_iet
            args.min_iet = max(args.min_iet, MIN_RES)
            if args.max_iet <= args.min_iet:
                raise ValueError("--max_iet must be greater than --min_iet.")
            min_iet = args.min_iet
            max_iet = args.max_iet
        num_bins = int(math.log10(max_iet / min_iet) * args.bins_per_decade + 1.5)
        xscale = "log"

    expected_shape = None if args.no_expected or args.expected_shape == 'none' else args.expected_shape
    picture_name = "iet_pdf" if args.distribution == "pdf" else "iet_hist"
    expected_shape_label = expected_shape or "none"
    if expected_shape_label == "gaussian":
        expected_shape_label = "normal"
    elif expected_shape_label == "log-normal":
        expected_shape_label = "lognormal"
    elif expected_shape_label in ("inverse-gaussian", "invgauss"):
        expected_shape_label = "inverse_gaussian"
    picture_name = f"{picture_name}_{expected_shape_label}"
    if args.linear:
        picture_name = f"{picture_name}_lin"

    plot_overall_iet_distribution(
        iet_spatial_grid, bins=num_bins, color='blue', 
        min=min_iet, max=max_iet, 
        expected_rate=args.rate,
        expected_total=args.expected_total,
        expected_std_dev=args.std_dev,
        expected_log_mean=args.log_mean,
        expected_log_std_dev=args.log_std_dev,
        expected_invgauss_mean=args.invgauss_mean,
        expected_invgauss_shape=args.invgauss_shape,
        expected_shape=expected_shape,
        distribution=args.distribution,
        xscale=xscale,
        save_path=str(fm.picture_base(
            picture_name,
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
