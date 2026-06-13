import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import counts
import file_manager as fm


def in_bounds_events(df, width, height):
    """Return events within the requested frame bounds."""
    in_bounds = (df["x"] >= 0) & (df["x"] < width) & (df["y"] >= 0) & (df["y"] < height)
    if not in_bounds.all():
        dropped = len(df) - int(in_bounds.sum())
        print(f"Warning: dropping {dropped:,}/{len(df):,} events outside the {width}x{height} frame")
        df = df[in_bounds].copy()
    return df


def binned_average_count_map(df, width, height, size):
    """
    Count events in size-by-size pixel bins, then divide by each bin's pixel area.

    Edge bins can be smaller than size-by-size when width or height is not evenly
    divisible by size.
    """
    if size <= 0:
        raise ValueError("--size must be positive.")

    bin_width = int(np.ceil(width / size))
    bin_height = int(np.ceil(height / size))
    count_map = np.zeros((bin_height, bin_width), dtype=float)

    if not df.empty:
        x_coords = np.clip(np.round(df["x"]).astype(int), 0, width - 1)
        y_coords = np.clip(np.round(df["y"]).astype(int), 0, height - 1)
        x_bins = x_coords // size
        y_bins = y_coords // size
        np.add.at(count_map, (y_bins, x_bins), 1)

    y_edges = np.arange(0, bin_height + 1) * size
    x_edges = np.arange(0, bin_width + 1) * size
    y_edges[-1] = height
    x_edges[-1] = width
    bin_areas = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]

    return count_map / bin_areas


def frame_windows(df, start_t, end_t, window, increment=None):
    """Build time windows and per-window event slices.
    
    Args:
        df: Event dataframe
        start_t: Start time in seconds
        end_t: End time in seconds
        window: Time window size in seconds
        increment: Time increment between windows in seconds. If None, defaults to window (non-overlapping).
    """
    if window <= 0:
        raise ValueError("--window must be positive.")
    if end_t <= start_t:
        raise ValueError("End time must be greater than start time.")
    
    if increment is None:
        increment = window
    if increment <= 0:
        raise ValueError("--increment must be positive.")

    windows = []
    t0 = start_t
    while t0 < end_t:
        t1 = min(t0 + window, end_t)
        windows.append((t0, t1, df[(df["t"] >= t0) & (df["t"] < t1)]))
        t0 += increment
    return windows


def save_temporal_count_video(
    df,
    output_path,
    *,
    width,
    height,
    size,
    start_t,
    end_t,
    window,
    fps,
    title,
    increment=None,
    show=True,
):
    """Create an MP4 of average event-count heatmaps over time. Returns maps and windows."""
    print(f"Preparing {window:g}s count windows from t={start_t:g}s to t={end_t:g}s...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    windows = frame_windows(df, start_t, end_t, window, increment=increment)
    maps = [binned_average_count_map(frame_df, width, height, size) for _, _, frame_df in windows]
    max_average = max(float(count_map.max()) for count_map in maps)
    print(f"Prepared {len(maps):,} frames. Max average events per pixel in any {size}x{size} bin: {max_average:g}")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
    im = ax.imshow(
        maps[0],
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=0,
        vmax=max_average if max_average > 0 else 1,
    )

    cbar = plt.colorbar(im, ax=ax, label=f"Average event count per pixel ({size}x{size} bins)", pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate", color="white")
    ax.set_ylabel("Y coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("black")
    title_obj = ax.set_title("", color="white", fontsize=14, pad=20)

    def update(frame_idx):
        t0, t1, frame_df = windows[frame_idx]
        im.set_data(maps[frame_idx])
        title_obj.set_text(f"{title}\nt={t0:.6g}s to {t1:.6g}s | Events: {len(frame_df):,}")
        print(
            f"\rTemporal Count Heatmap | frame {frame_idx + 1:,}/{len(maps):,} | "
            f"t={t0:.6g}s to {t1:.6g}s | events={len(frame_df):,}",
            end="",
            flush=True,
        )
        return [im, title_obj]

    ani = animation.FuncAnimation(fig, update, frames=len(maps), interval=1000 / fps, blit=True)
    plt.tight_layout()

    print(f"Saving video to {output_path}...")
    ani.save(output_path, writer="ffmpeg", fps=fps)
    print()
    print("Video saved successfully.")

    if show:
        plt.show()
    plt.close(fig)

    return maps, windows


def resolve_end_time(df, start_t, duration):
    if duration is None or np.isinf(duration):
        return float(df["t"].max())
    return start_t + duration


def save_temporal_counts_data(maps, windows, output_path, *, size, window, width, height):
    """Save temporal count maps and metadata as numpy archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract time information
    times = np.array([[t0, t1] for t0, t1, _ in windows], dtype=float)
    
    # Stack maps into a single 3D array
    maps_array = np.stack(maps, axis=0)
    
    # Save as npz
    np.savez(
        output_path,
        maps=maps_array,
        times=times,
        size=size,
        window=window,
        width=width,
        height=height,
    )
    print(f"Saved temporal counts data to {output_path}")


def save_delta_temporal_count_video(
    maps,
    windows,
    output_path,
    *,
    width,
    height,
    size,
    window,
    fps,
    title,
    show=True,
):
    """Create an MP4 showing the change in average event-count over time. Returns delta maps."""
    print(f"Preparing delta maps (change per window)...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate delta: (current - previous) / window
    delta_maps = []
    for i, count_map in enumerate(maps):
        if i == 0:
            # First frame: delta is 0
            delta_map = np.zeros_like(count_map)
        else:
            delta_map = (count_map - maps[i - 1]) / window
        delta_maps.append(delta_map)

    # Find symmetric range for diverging colormap
    max_abs_delta = max(float(np.abs(delta_map).max()) for delta_map in delta_maps) if delta_maps else 1.0
    if max_abs_delta == 0:
        max_abs_delta = 1.0

    print(f"Prepared {len(delta_maps):,} frames. Max absolute delta: {max_abs_delta:g}")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
    im = ax.imshow(
        delta_maps[0],
        cmap="RdBu_r",
        origin="upper",
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
    )

    cbar = plt.colorbar(im, ax=ax, label=f"Change in average count per pixel / {size}x{size} bin / window", pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate", color="white")
    ax.set_ylabel("Y coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("black")
    title_obj = ax.set_title("", color="white", fontsize=14, pad=20)

    def update(frame_idx):
        t0, t1, _ = windows[frame_idx]
        im.set_data(delta_maps[frame_idx])
        title_obj.set_text(f"{title}\nt={t0:.6g}s to {t1:.6g}s | Delta (change per window)")
        print(
            f"\rDelta Heatmap | frame {frame_idx + 1:,}/{len(delta_maps):,} | "
            f"t={t0:.6g}s to {t1:.6g}s",
            end="",
            flush=True,
        )
        return [im, title_obj]

    ani = animation.FuncAnimation(fig, update, frames=len(delta_maps), interval=1000 / fps, blit=True)
    plt.tight_layout()

    print(f"Saving delta video to {output_path}...")
    ani.save(output_path, writer="ffmpeg", fps=fps)
    print()
    print("Delta video saved successfully.")

    if show:
        plt.show()
    plt.close(fig)

    return delta_maps


def save_delta_temporal_counts_data(delta_maps, windows, output_path, *, size, window, width, height):
    """Save delta temporal count maps and metadata as numpy archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract time information
    times = np.array([[t0, t1] for t0, t1, _ in windows], dtype=float)
    
    # Stack maps into a single 3D array
    delta_maps_array = np.stack(delta_maps, axis=0)
    
    # Save as npz
    np.savez(
        output_path,
        delta_maps=delta_maps_array,
        times=times,
        size=size,
        window=window,
        width=width,
        height=height,
    )
    print(f"Saved delta temporal counts data to {output_path}")


def save_velocity_video(
    maps,
    windows,
    output_path,
    *,
    width,
    height,
    size,
    window,
    patch_size,
    increment=None,
    fps,
    title,
    show=True,
):
    """
    Apply Local Optical Flow Constraint Residual (LOFCR) estimation to cancel egomotion
    and render the output animation stream. Returns velocity/lofcr maps.
    
    Args:
        increment: Time increment between windows in seconds. Used for display purposes.
    """
    print(f"Calculating Velocity maps (patch={patch_size}x{patch_size})...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames = len(maps)
    bin_height, bin_width = maps[0].shape
    lofcr_maps = []

    # Calculate temporal gradients (symmetric central differences where possible)
    m_t_all = []
    for i in range(num_frames):
        if num_frames <= 1:
            mt = np.zeros_like(maps[0])
        elif i == 0:
            mt = (maps[1] - maps[0]) / window
        elif i == num_frames - 1:
            mt = (maps[-1] - maps[-2]) / window
        else:
            mt = (maps[i + 1] - maps[i - 1]) / (2 * window)
        m_t_all.append(mt)

    half_p = patch_size // 2

    for i in range(num_frames):
        count_map = maps[i]
        mt = m_t_all[i]

        # Calculate spatial gradients for the current binned map frame
        my, mx = np.gradient(count_map)

        # Precompute element-wise products for the linear system components
        Ixx = mx * mx
        Iyy = my * my
        Ixy = mx * my
        Ixt = mx * mt
        Iyt = my * mt

        residual_map = np.zeros_like(count_map)

        # Slide across each individual macro-block spatial bin configuration
        for r in range(bin_height):
            for c in range(bin_width):
                r_start = max(0, r - half_p)
                r_end = min(bin_height, r + half_p + 1)
                c_start = max(0, c - half_p)
                c_end = min(bin_width, c + half_p + 1)

                # Compute patch sums for our normal matrix configuration
                Sxx = np.sum(Ixx[r_start:r_end, c_start:c_end])
                Syy = np.sum(Iyy[r_start:r_end, c_start:c_end])
                Sxy = np.sum(Ixy[r_start:r_end, c_start:c_end])
                Sxt = np.sum(Ixt[r_start:r_end, c_start:c_end])
                Syt = np.sum(Iyt[r_start:r_end, c_start:c_end])

                # Use analytical Cramer's rule to extract local velocity parameterization
                det = Sxx * Syy - Sxy * Sxy
                if det > 1e-5:
                    vx = (Syt * Sxy - Sxt * Syy) / det
                    vy = (Sxy * Sxt - Sxx * Syt) / det
                else:
                    vx, vy = 0.0, 0.0

                # Compute the structural constraint deviation at the focal macro-block node
                residual_map[r, c] = np.abs(mx[r, c] * vx + my[r, c] * vy + mt[r, c])

        lofcr_maps.append(residual_map)

    max_residual = max(float(rmap.max()) for rmap in lofcr_maps) if lofcr_maps else 1.0
    if max_residual == 0:
        max_residual = 1.0

    print(f"Prepared {len(lofcr_maps):,} Velocity frames. Max Residual Value: {max_residual:g}")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
    im = ax.imshow(
        lofcr_maps[0],
        cmap="inferno",
        origin="upper",
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=0,
        vmax=max_residual * 0.6,  # Squeeze the display threshold dynamically to make target bounds visually distinct
    )

    cbar = plt.colorbar(im, ax=ax, label=f"Velocity Deviation Anomaly Intensity ({patch_size}x{patch_size} Patch Setting)", pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate", color="white")
    ax.set_ylabel("Y coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("black")
    title_obj = ax.set_title("", color="white", fontsize=14, pad=20)

    def update(frame_idx):
        t0, t1, _ = windows[frame_idx]
        im.set_data(lofcr_maps[frame_idx])
        title_obj.set_text(f"{title}\nt={t0:.6g}s to {t1:.6g}s | Velocity Egomotion Suppression Matrix")
        print(
            f"\rVelocity Heatmap | frame {frame_idx + 1:,}/{len(lofcr_maps):,} | "
            f"t={t0:.6g}s to {t1:.6g}s",
            end="",
            flush=True,
        )
        return [im, title_obj]

    ani = animation.FuncAnimation(fig, update, frames=len(lofcr_maps), interval=1000 / fps, blit=True)
    plt.tight_layout()

    print(f"Saving Velocity tracking video output to {output_path}...")
    ani.save(output_path, writer="ffmpeg", fps=fps)
    print()
    print("Velocity video file written successfully.")

    if show:
        plt.show()
    plt.close(fig)

    return lofcr_maps


def save_velocity_data(lofcr_maps, windows, output_path, *, size, window, patch_size, width, height, increment=None):
    """Save velocity/LOFCR tracking data and parameter sets to disk as numpy archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    times = np.array([[t0, t1] for t0, t1, _ in windows], dtype=float)
    lofcr_maps_array = np.stack(lofcr_maps, axis=0)
    
    np.savez(
        output_path,
        lofcr_maps=lofcr_maps_array,
        times=times,
        size=size,
        window=window,
        patch_size=patch_size,
        width=width,
        height=height,
        increment=increment,
    )
    print(f"Saved velocity tracking data matrices to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a video of binned event-count heatmaps over time."
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Dataset duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --dataset is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object event files.")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename for the events CSV file.")
    parser.add_argument("--line", type=int, default=None, help="Use line-filtered events file with the specified line number.")

    parser.add_argument("--window", type=float, required=True, help="Time window size in seconds for each heatmap frame.")
    parser.add_argument("--size", type=int, default=8, help="Square spatial bin size in pixels (default: 8 for 8x8 bins).")
    parser.add_argument("--video", type=float, default=float("inf"), help="Video duration in seconds (default: full available range).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds for the video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for the MP4 video.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the animation window.")
    
    # Added spatial window size option for LOFCR tracking estimation
    parser.add_argument("--patch_size", type=int, default=3, help="Odd integer configuration for spatial calculation neighborhoods (e.g., 3 for a 3x3 patch layout).")
    parser.add_argument("--increment", type=float, default=None, help="Time increment between consecutive windows in seconds. Defaults to window size for non-overlapping windows. Use smaller values for better time resolution.")

    args = parser.parse_args()

    if args.patch_size % 2 == 0 or args.patch_size < 1:
        raise ValueError("--patch_size argument configuration must represent a positive odd integer designation.")

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

    df = counts.load_events_csv(csv_path)
    df = counts.filter_by_polarity(df, args.polarity)
    df = in_bounds_events(df, args.width, args.height)
    end_t = resolve_end_time(df, args.start, args.video)
    df = df[(df["t"] >= args.start) & (df["t"] < end_t)].copy()

    if df.empty:
        print("ERROR: No events to process after filtering")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        base_stem, output_polarity, output_line = counts.split_count_stem(
            csv_path.stem,
            polarity=args.polarity,
            line=args.line,
        )
        output_path = fm.video_file(
            f"temporal_counts_{args.window:g}s_{args.size}px",
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=args.rate,
            duration=args.duration,
            stem=base_stem,
            slice_name=slice_name,
            polarity=output_polarity,
            line=output_line,
            create_parent=True,
        )

    title = f"Average Event Count Heatmap | {csv_path.name}"
    if args.polarity:
        title += f" | Polarity: {args.polarity}"

    maps, windows = save_temporal_count_video(
        df,
        output_path,
        width=args.width,
        height=args.height,
        size=args.size,
        start_t=args.start,
        end_t=end_t,
        window=args.window,
        increment=args.increment,
        fps=args.fps,
        title=title,
        show=not args.no_show,
    )

    # Save temporal counts data
    data_output_path = Path(str(output_path).replace(".mp4", ".npz"))
    save_temporal_counts_data(
        maps,
        windows,
        data_output_path,
        size=args.size,
        window=args.window,
        width=args.width,
        height=args.height,
    )

    # Generate and save delta video
    delta_output_path = Path(str(output_path).replace("temporal_counts", "delta_temporal_counts"))
    delta_title = title.replace("Average Event Count", "Delta (Change in Event Count)")
    delta_maps = save_delta_temporal_count_video(
        maps,
        windows,
        delta_output_path,
        width=args.width,
        height=args.height,
        size=args.size,
        window=args.window,
        fps=args.fps,
        title=delta_title,
        show=not args.no_show,
    )

    # Save delta temporal counts data
    delta_data_output_path = Path(str(delta_output_path).replace(".mp4", ".npz"))
    save_delta_temporal_counts_data(
        delta_maps,
        windows,
        delta_data_output_path,
        size=args.size,
        window=args.window,
        width=args.width,
        height=args.height,
    )

    # Generate and save velocity/egomotion suppression layer video output
    velocity_output_path = Path(str(output_path).replace("temporal_counts", f"velocity_counts_{args.patch_size}px"))
    velocity_title = title.replace("Average Event Count", "Velocity Object Motion Tracking")
    velocity_maps = save_velocity_video(
        maps,
        windows,
        velocity_output_path,
        width=args.width,
        height=args.height,
        size=args.size,
        window=args.window,
        patch_size=args.patch_size,
        increment=args.increment,
        fps=args.fps,
        title=velocity_title,
        show=not args.no_show,
    )

    # Save tracking analysis parameters matrix archive
    velocity_data_output_path = Path(str(velocity_output_path).replace(".mp4", ".npz"))
    save_velocity_data(
        velocity_maps,
        windows,
        velocity_data_output_path,
        size=args.size,
        window=args.window,
        patch_size=args.patch_size,
        width=args.width,
        height=args.height,
        increment=args.increment,
    )