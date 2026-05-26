import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def load_velocity_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"start_t", "end_t", "center_t", "vx_pixels_per_s", "vy_pixels_per_s"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required velocity CSV columns: {sorted(missing_cols)}")
    return df.sort_values("start_t").reset_index(drop=True)


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


def attach_velocity_windows(events, velocities):
    events = events.sort_values("t").reset_index(drop=True)
    velocity_cols = [
        col
        for col in [
            "window_index",
            "start_t",
            "end_t",
            "center_t",
            "vx_pixels_per_s",
            "vy_pixels_per_s",
            "speed_pixels_per_s",
        ]
        if col in velocities.columns
    ]
    merged = pd.merge_asof(
        events,
        velocities[velocity_cols].sort_values("start_t"),
        left_on="t",
        right_on="start_t",
        direction="backward",
    )
    return merged[merged["t"] < merged["end_t"]].copy()


def _signed_step(values):
    return np.sign(values).astype(int)


def add_neighbor_predictions(events, width, height, axis="x", dx=None, dy=None, min_speed=1e-9):
    predicted = events.copy()

    if dx is None and dy is None:
        if axis == "x":
            speed_component = predicted["vx_pixels_per_s"].abs()
            neighbor_dx = _signed_step(predicted["vx_pixels_per_s"].fillna(0.0))
            neighbor_dy = np.zeros(len(predicted), dtype=int)
        elif axis == "y":
            speed_component = predicted["vy_pixels_per_s"].abs()
            neighbor_dx = np.zeros(len(predicted), dtype=int)
            neighbor_dy = _signed_step(predicted["vy_pixels_per_s"].fillna(0.0))
        else:
            use_x = predicted["vx_pixels_per_s"].abs() >= predicted["vy_pixels_per_s"].abs()
            neighbor_dx = np.where(use_x, _signed_step(predicted["vx_pixels_per_s"].fillna(0.0)), 0)
            neighbor_dy = np.where(use_x, 0, _signed_step(predicted["vy_pixels_per_s"].fillna(0.0)))
            speed_component = np.where(
                use_x,
                predicted["vx_pixels_per_s"].abs(),
                predicted["vy_pixels_per_s"].abs(),
            )
        distance_pixels = 1.0
    else:
        dx = 0 if dx is None else int(dx)
        dy = 0 if dy is None else int(dy)
        if dx == 0 and dy == 0:
            raise ValueError("At least one of dx or dy must be non-zero.")
        neighbor_dx = np.full(len(predicted), dx, dtype=int)
        neighbor_dy = np.full(len(predicted), dy, dtype=int)
        distance_pixels = float(np.hypot(dx, dy))
        speed_component = (
            predicted["vx_pixels_per_s"] * dx + predicted["vy_pixels_per_s"] * dy
        ) / distance_pixels

    predicted["event_x"] = np.rint(predicted["x"]).astype(int)
    predicted["event_y"] = np.rint(predicted["y"]).astype(int)
    predicted["neighbor_dx"] = neighbor_dx
    predicted["neighbor_dy"] = neighbor_dy
    predicted["target_x"] = predicted["event_x"] + predicted["neighbor_dx"]
    predicted["target_y"] = predicted["event_y"] + predicted["neighbor_dy"]
    predicted["velocity_component_pixels_per_s"] = speed_component

    valid = (
        np.isfinite(predicted["velocity_component_pixels_per_s"])
        & (predicted["velocity_component_pixels_per_s"] > min_speed)
        & ((predicted["neighbor_dx"] != 0) | (predicted["neighbor_dy"] != 0))
        & (predicted["target_x"] >= 0)
        & (predicted["target_x"] < width)
        & (predicted["target_y"] >= 0)
        & (predicted["target_y"] < height)
    )
    predicted = predicted[valid].copy()
    predicted["predicted_dt"] = distance_pixels / predicted["velocity_component_pixels_per_s"]
    predicted["predicted_t"] = predicted["t"] + predicted["predicted_dt"]
    return predicted


def pixel_time_lookup(events):
    lookup = {}
    rounded = events.copy()
    rounded["event_x"] = np.rint(rounded["x"]).astype(int)
    rounded["event_y"] = np.rint(rounded["y"]).astype(int)
    for (x, y), group in rounded.groupby(["event_x", "event_y"]):
        lookup[(int(x), int(y))] = np.sort(group["t"].to_numpy(dtype=float))
    return lookup


def add_actual_neighbor_times(predictions, events):
    lookup = pixel_time_lookup(events)
    actual_times = np.full(len(predictions), np.nan, dtype=float)

    for out_idx, (_, row) in enumerate(predictions.iterrows()):
        times = lookup.get((int(row["target_x"]), int(row["target_y"])))
        if times is None or len(times) == 0:
            continue
        predicted_t = float(row["predicted_t"])
        next_idx = np.searchsorted(times, predicted_t, side="left")
        candidates = []
        if next_idx > 0:
            candidates.append(times[next_idx - 1])
        if next_idx < len(times):
            candidates.append(times[next_idx])
        if candidates:
            actual_times[out_idx] = min(candidates, key=lambda event_t: abs(event_t - predicted_t))

    predictions = predictions.copy()
    predictions["actual_t"] = actual_times
    predictions = predictions[predictions["actual_t"].notna()].copy()
    predictions["actual_dt"] = predictions["actual_t"] - predictions["t"]
    predictions["residual_t"] = predictions["actual_dt"] - predictions["predicted_dt"]
    return predictions


def prediction_residuals(events, velocities, width, height, axis="x", dx=None, dy=None, min_speed=1e-9):
    events_with_velocity = attach_velocity_windows(events, velocities)
    predictions = add_neighbor_predictions(
        events_with_velocity,
        width,
        height,
        axis=axis,
        dx=dx,
        dy=dy,
        min_speed=min_speed,
    )
    residuals = add_actual_neighbor_times(predictions, events)
    columns = [
        "window_index",
        "start_t",
        "end_t",
        "center_t",
        "event_x",
        "event_y",
        "t",
        "vx_pixels_per_s",
        "vy_pixels_per_s",
        "neighbor_dx",
        "neighbor_dy",
        "target_x",
        "target_y",
        "velocity_component_pixels_per_s",
        "predicted_dt",
        "predicted_t",
        "actual_t",
        "actual_dt",
        "residual_t",
    ]
    return residuals[[col for col in columns if col in residuals.columns]]


def plot_residual_distribution(residuals, output_base, bins=100, min_residual=None, max_residual=None, show=True):
    valid = residuals["residual_t"].to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        raise ValueError("No valid residuals to plot.")
    mean_residual = float(np.mean(valid))

    speeds = residuals["velocity_component_pixels_per_s"].to_numpy(dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else float("nan")

    if min_residual is not None or max_residual is not None:
        lower = np.min(valid) if min_residual is None else min_residual
        upper = np.max(valid) if max_residual is None else max_residual
        if upper <= lower:
            raise ValueError("Residual plot range is empty; check --min_residual and --max_residual.")
        valid = valid[(valid >= lower) & (valid <= upper)]
        if len(valid) == 0:
            raise ValueError("No residuals remain inside the requested plot range.")
        hist_range = (lower, upper)
    else:
        hist_range = None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=bins, range=hist_range, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="orange", linestyle="--", linewidth=2, label="perfect prediction")
    ax.axvline(
        mean_residual,
        color="crimson",
        linestyle="-",
        linewidth=2,
        label=f"mean residual: {mean_residual:.3g}s",
    )
    ax.set_title(
        "Neighbor Event Timing Residuals\n"
        f"Mean projected speed: {mean_speed:.3g} pixels/s"
    )
    ax.set_xlabel("actual_dt - predicted_dt (seconds)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    hist_values, bin_edges = np.histogram(valid, bins=bins, range=hist_range)
    pd.DataFrame(
        {
            "bin": np.arange(len(hist_values)),
            "left": bin_edges[:-1],
            "right": bin_edges[1:],
            "count": hist_values,
        }
    ).to_csv(str(output_base) + ".csv", index=False)
    if show:
        plt.show()
    plt.close(fig)


def neighbor_label(args):
    if args.dx is not None or args.dy is not None:
        return f"dx{0 if args.dx is None else args.dx}_dy{0 if args.dy is None else args.dy}"
    return f"auto_{args.axis}"


def resolve_paths(args, source, dataset, slice_name, event_stem):
    velocity_name = f"center_of_mass_velocity_{fm.seconds_label(args.window)}"
    velocity_path = fm.find_track_file(
        velocity_name,
        filename=args.velocity_filename,
        data_root=args.data_root,
        source=source,
        dataset=dataset,
        rate=args.rate,
        duration=args.duration,
        stem=event_stem,
        slice_name=slice_name,
        polarity=args.polarity,
        line=args.line,
    )

    output_name = f"neighbor_time_residual_{fm.seconds_label(args.window)}_{neighbor_label(args)}"
    if args.output:
        residual_output = Path(args.output)
    else:
        residual_output = fm.track_file(
            output_name,
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

    if args.plot_output:
        plot_output = Path(args.plot_output)
    else:
        plot_output = fm.picture_base(
            f"{output_name}_hist",
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

    return velocity_path, residual_output, plot_output


def main():
    parser = argparse.ArgumentParser(
        description="Predict neighboring-pixel event time from COM velocity and graph timing residuals."
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
    parser.add_argument("--line", type=int, default=None, help="Use line-filtered events and velocity files.")
    parser.add_argument("--filename", type=str, default=None, help="Explicit events CSV path.")
    parser.add_argument("--velocity_filename", type=str, default=None, help="Explicit COM velocity CSV path.")
    parser.add_argument("--window", type=float, default=0.001, help="Velocity window in seconds.")
    parser.add_argument("--axis", choices=["x", "y", "dominant"], default="x", help="Auto neighbor direction when dx/dy are not set.")
    parser.add_argument("--dx", type=int, default=None, help="Explicit neighbor x offset.")
    parser.add_argument("--dy", type=int, default=None, help="Explicit neighbor y offset.")
    parser.add_argument("--min_speed", type=float, default=1e-9, help="Minimum projected speed to make a prediction.")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins.")
    parser.add_argument("--min_residual", type=float, default=-0.001, help="Minimum residual to plot.")
    parser.add_argument("--max_residual", type=float, default=None, help="Maximum residual to plot.")
    parser.add_argument("--output", type=str, default=None, help="Explicit output CSV for per-event residuals.")
    parser.add_argument("--plot_output", type=str, default=None, help="Explicit plot base path, without extension.")
    parser.add_argument("--no_show", action="store_true", help="Suppress showing the histogram.")
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
    event_stem = Path(filename).stem
    velocity_path, residual_output, plot_output = resolve_paths(args, source, dataset, slice_name, event_stem)

    events = load_events_csv(filename)
    events = filter_by_polarity(events, args.polarity)
    events = filter_in_bounds(events, args.width, args.height)
    velocities = load_velocity_csv(velocity_path)

    residuals = prediction_residuals(
        events,
        velocities,
        args.width,
        args.height,
        axis=args.axis,
        dx=args.dx,
        dy=args.dy,
        min_speed=args.min_speed,
    )
    if residuals.empty:
        raise ValueError("No neighbor timing residuals were generated. Check velocity direction, line, and event density.")

    residual_output.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(residual_output, index=False)
    plot_residual_distribution(
        residuals,
        plot_output,
        bins=args.bins,
        min_residual=args.min_residual,
        max_residual=args.max_residual,
        show=not args.no_show,
    )

    print(f"Neighbor timing residuals saved to {residual_output}")
    print(f"Residual histogram saved to {plot_output}.png")
    print(
        f"Generated {len(residuals):,} residuals; "
        f"mean={residuals['residual_t'].mean():.6g}s, "
        f"std={residuals['residual_t'].std():.6g}s."
    )


if __name__ == "__main__":
    main()
