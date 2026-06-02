import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import file_manager as fm


def load_events_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"x", "y", "t"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_cols)}")
    return df


def filter_by_polarity(df, polarity):
    if polarity is None:
        return df
    if "p" not in df.columns:
        raise ValueError("Polarity column 'p' not found in CSV.")
    polarity_value = {"ON": 1, "OFF": 0}[polarity]
    return df[df["p"] == polarity_value].copy()


def filter_in_bounds(df, width, height):
    in_bounds = (df["x"] >= 0) & (df["x"] < width) & (df["y"] >= 0) & (df["y"] < height)
    dropped = len(df) - int(in_bounds.sum())
    if dropped:
        print(f"Warning: dropping {dropped:,}/{len(df):,} events outside the {width}x{height} frame")
    return df[in_bounds].copy()


def center_of_mass_snapshots(df, window=0.001, start=None, end=None, include_empty=True):
    if window <= 0:
        raise ValueError("window must be positive.")
    if df.empty:
        raise ValueError("No events available for center-of-mass tracking.")

    start = float(df["t"].min()) if start is None else float(start)
    if end is None:
        max_t = float(df["t"].max())
        end = max_t + max(np.finfo(float).eps * max(abs(max_t), 1.0), window * 1e-9)
    else:
        end = float(end)
    if end <= start:
        end = start + window

    active = df[(df["t"] >= start) & (df["t"] < end)].copy()
    if active.empty:
        raise ValueError("No events remain inside the requested tracking time range.")

    num_windows = max(1, int(np.ceil((end - start) / window)))
    active["window_index"] = np.floor((active["t"] - start) / window).astype(int)
    active = active[(active["window_index"] >= 0) & (active["window_index"] < num_windows)]

    grouped = active.groupby("window_index", sort=True)
    snapshots = grouped.agg(
        event_count=("t", "size"),
        com_x=("x", "mean"),
        com_y=("y", "mean"),
        first_event_t=("t", "min"),
        last_event_t=("t", "max"),
    ).reset_index()

    if include_empty:
        all_windows = pd.DataFrame({"window_index": np.arange(num_windows, dtype=int)})
        snapshots = all_windows.merge(snapshots, on="window_index", how="left")
        snapshots["event_count"] = snapshots["event_count"].fillna(0).astype(int)

    snapshots["start_t"] = start + snapshots["window_index"] * window
    snapshots["end_t"] = np.minimum(snapshots["start_t"] + window, end)
    snapshots["center_t"] = 0.5 * (snapshots["start_t"] + snapshots["end_t"])

    columns = [
        "window_index",
        "start_t",
        "end_t",
        "center_t",
        "event_count",
        "com_x",
        "com_y",
        "first_event_t",
        "last_event_t",
    ]
    return snapshots[columns]


def center_of_mass_velocities(snapshots):
    velocities = snapshots.copy()
    velocities["prev_center_t"] = velocities["center_t"].shift(1)
    velocities["prev_com_x"] = velocities["com_x"].shift(1)
    velocities["prev_com_y"] = velocities["com_y"].shift(1)
    velocities["next_center_t"] = velocities["center_t"].shift(-1)
    velocities["next_com_x"] = velocities["com_x"].shift(-1)
    velocities["next_com_y"] = velocities["com_y"].shift(-1)
    velocities["prev_dt"] = velocities["center_t"] - velocities["prev_center_t"]
    velocities["next_dt"] = velocities["next_center_t"] - velocities["center_t"]

    prev_valid = (
        velocities["prev_dt"].notna()
        & (velocities["prev_dt"] > 0)
        & velocities["com_x"].notna()
        & velocities["com_y"].notna()
        & velocities["prev_com_x"].notna()
        & velocities["prev_com_y"].notna()
    )
    next_valid = (
        velocities["next_dt"].notna()
        & (velocities["next_dt"] > 0)
        & velocities["com_x"].notna()
        & velocities["com_y"].notna()
        & velocities["next_com_x"].notna()
        & velocities["next_com_y"].notna()
    )
    prev_vx = (velocities["com_x"] - velocities["prev_com_x"]) / velocities["prev_dt"]
    prev_vy = (velocities["com_y"] - velocities["prev_com_y"]) / velocities["prev_dt"]
    next_vx = (velocities["next_com_x"] - velocities["com_x"]) / velocities["next_dt"]
    next_vy = (velocities["next_com_y"] - velocities["com_y"]) / velocities["next_dt"]
    prev_vx = prev_vx.where(prev_valid)
    prev_vy = prev_vy.where(prev_valid)
    next_vx = next_vx.where(next_valid)
    next_vy = next_vy.where(next_valid)

    velocities["dt"] = pd.concat(
        [
            velocities["prev_dt"].where(prev_valid),
            velocities["next_dt"].where(next_valid),
        ],
        axis=1,
    ).mean(axis=1)
    velocities["vx_pixels_per_s"] = np.nan
    velocities["vy_pixels_per_s"] = np.nan
    velocities["vx_pixels_per_s"] = pd.concat([prev_vx, next_vx], axis=1).mean(axis=1)
    velocities["vy_pixels_per_s"] = pd.concat([prev_vy, next_vy], axis=1).mean(axis=1)
    velocities["speed_pixels_per_s"] = np.hypot(
        velocities["vx_pixels_per_s"],
        velocities["vy_pixels_per_s"],
    )

    columns = [
        "window_index",
        "start_t",
        "end_t",
        "center_t",
        "event_count",
        "com_x",
        "com_y",
        "prev_center_t",
        "prev_com_x",
        "prev_com_y",
        "dt",
        "vx_pixels_per_s",
        "vy_pixels_per_s",
        "speed_pixels_per_s",
    ]
    return velocities[columns]


def window_track_name(base_name, window):
    return f"{base_name}_{fm.seconds_label(window)}"


def resolve_output_paths(args, source, dataset, slice_name, event_stem):
    if args.center_output:
        center_output = Path(args.center_output)
    else:
        center_output = fm.track_file(
            window_track_name("center_of_mass", args.window),
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=args.rate,
            duration=args.duration,
            stem=event_stem,
            slice_name=slice_name,
            polarity=args.polarity,
            line=args.line,
            create_parent=True,
        )

    if args.velocity_output:
        velocity_output = Path(args.velocity_output)
    else:
        velocity_output = fm.track_file(
            window_track_name("center_of_mass_velocity", args.window),
            data_root=args.data_root,
            source=source,
            dataset=dataset,
            rate=args.rate,
            duration=args.duration,
            stem=event_stem,
            slice_name=slice_name,
            polarity=args.polarity,
            line=args.line,
            create_parent=True,
        )

    return center_output, velocity_output


def main():
    parser = argparse.ArgumentParser(
        description="Track event center-of-mass position and velocity over fixed time windows."
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT)
    parser.add_argument("--source", choices=fm.SOURCES, default=None)
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None)
    parser.add_argument("--slice", dest="slice_name", type=str, default=None)
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None)
    parser.add_argument("--line", type=int, default=None, help="Use line-filtered events file with this line number.")
    parser.add_argument("--filename", type=str, default=None, help="Explicit events CSV path.")
    parser.add_argument("--window", type=float, default=0.001, help="Tracking window in seconds.")
    parser.add_argument("--start", type=float, default=None, help="Optional tracking start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="Optional tracking end time in seconds.")
    parser.add_argument("--drop_empty", action="store_true", help="Drop windows with no events from the output CSVs.")
    parser.add_argument("--center_output", type=str, default=None, help="Explicit output CSV for COM snapshots.")
    parser.add_argument("--velocity_output", type=str, default=None, help="Explicit output CSV for COM velocities.")
    args = parser.parse_args()

    source = args.source or fm.SOURCE_NOISE
    filename = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=source,
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

    events = load_events_csv(filename)
    events = filter_by_polarity(events, args.polarity)
    events = filter_in_bounds(events, args.width, args.height)

    snapshots = center_of_mass_snapshots(
        events,
        window=args.window,
        start=args.start,
        end=args.end,
        include_empty=not args.drop_empty,
    )
    velocities = center_of_mass_velocities(snapshots)

    event_stem = Path(filename).stem
    center_output, velocity_output = resolve_output_paths(args, source, dataset, slice_name, event_stem)
    center_output.parent.mkdir(parents=True, exist_ok=True)
    velocity_output.parent.mkdir(parents=True, exist_ok=True)

    snapshots.to_csv(center_output, index=False)
    velocities.to_csv(velocity_output, index=False)

    valid_velocity_count = int(velocities["speed_pixels_per_s"].notna().sum())
    print(f"Center-of-mass snapshots saved to {center_output}")
    print(f"Center-of-mass velocities saved to {velocity_output}")
    print(
        f"Tracked {len(snapshots):,} windows at {args.window:g}s "
        f"with {valid_velocity_count:,} valid velocity estimates."
    )


if __name__ == "__main__":
    main()
