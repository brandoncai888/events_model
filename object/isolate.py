import pandas as pd
from pathlib import Path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import file_manager as fm


def get_line_pixels(x0, y0, x1, y1, width, height):
    """
    Returns the in-bounds integer pixel coordinates on a line segment.
    Uses Bresenham's line algorithm.
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


def line_keep_pixels(x0, y0, x1, y1, width, height, line_radius=0):
    """Returns the set of pixels to keep around a line with optional radius."""
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
    return keep_pixels


def next_line_label(output_dir, base_stem):
    """Find the next available line label for this stem."""
    output_dir = Path(output_dir)
    pattern = re.compile(rf"^{re.escape(base_stem)}_line(\d+)\.csv$")
    existing = []
    if output_dir.exists():
        for path in output_dir.glob(f"{base_stem}_line*"):
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


def save_line_mask_image(image_filename, keep_pixels, width, height, keep_line, line_radius, show=False):
    image_filename = Path(image_filename)
    image_filename.parent.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in keep_pixels:
        mask[y, x] = 1

    x0, y0, x1, y1 = keep_line
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.imshow(mask, cmap="gray", origin="upper", interpolation="nearest", vmin=0, vmax=1)
    ax.plot([x0, x1], [y0, y1], color="red", linewidth=0.8, alpha=0.8)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Kept pixels | radius={line_radius}", color="white")
    ax.set_xlabel("X coordinate", color="white")
    ax.set_ylabel("Y coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("black")
    plt.tight_layout()
    fig.savefig(image_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Line mask image saved to {image_filename}")


def save_line_info(
    info_filename,
    *,
    line_label,
    keep_line,
    line_radius,
    width,
    height,
    input_filename,
    output_filenames,
    line_mask_filename,
    total_events,
    kept_events,
    kept_event_pixels,
):
    """Save metadata about the line filter."""
    x0, y0, x1, y1 = keep_line
    lines = [
        f"line_label: {line_label}",
        f"x0: {x0}",
        f"y0: {y0}",
        f"x1: {x1}",
        f"y1: {y1}",
        f"line_radius: {line_radius}",
        f"sensor_width: {width}",
        f"sensor_height: {height}",
        f"total_events: {total_events}",
        f"kept_events: {kept_events}",
        f"kept_event_pixels: {kept_event_pixels}",
        f"input_events: {input_filename}",
    ]
    for polarity, output_file in output_filenames.items():
        lines.append(f"output_events_{polarity}: {output_file}")
    lines.append(f"line_mask_image: {line_mask_filename}")

    Path(info_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Line metadata saved to {info_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze event counts per pixel from a CSV file.")
    parser.add_argument("--filename", type=str, default=None, help="Optional explicit CSV path containing event data with columns 'x', 'y', and 't'.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=fm.SOURCE_OBJECT, help="Data source folder (default: object).")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name, for example 45Hz.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--mintime", type=float, default=0.0, help="Minimum time for filtering events.")
    parser.add_argument("--maxtime", type=float, default=float('inf'), help="Maximum time for filtering events.")
    parser.add_argument("--keep_line", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Keep events only along this pixel line and clear all other pixels.")
    parser.add_argument("--line_radius", type=int, default=0, help="Optional radius around --keep_line pixels to keep.")
    parser.add_argument("--show_line_mask", action="store_true", help="Display the generated line mask image after saving it.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    args = parser.parse_args()

    input_file = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
    )
    context = fm.context_from_path(
        input_file,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
        slice_name=args.slice_name,
    )
    output_slice = fm.slice_from_window(args.mintime, args.maxtime, slice_name=context["slice_name"])
    base_stem = Path(input_file).stem
    line_label = None
    line_kwarg = {}
    kept_pixels = None

    # Determine line label if requested
    if args.keep_line is not None:
        events_dir = fm.artifact_dir(
            fm.ARTIFACT_EVENTS,
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            slice_name=output_slice,
            create=True,
        )
        line_base_file = fm.events_file(
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            stem=base_stem,
            slice_name=output_slice,
        )
        line_label = next_line_label(events_dir, line_base_file.stem)
        line_kwarg = {"line": line_label}
        print(f"Using line label: {line_label}")

    # Load the full CSV file with time and line filtering
    df = pd.read_csv(input_file)
    total_events = len(df)
    df = df[df["t"] >= args.mintime]
    df = df[df["t"] < args.maxtime]

    # Apply line filtering if requested
    if args.keep_line is not None:
        x0, y0, x1, y1 = args.keep_line
        kept_pixels = line_keep_pixels(x0, y0, x1, y1, args.width, args.height, line_radius=args.line_radius)
        kept_pixel_codes = {y * args.width + x for x, y in kept_pixels}
        event_pixel_codes = df["y"].astype(np.int64) * args.width + df["x"].astype(np.int64)
        df = df[event_pixel_codes.isin(kept_pixel_codes)]
        print(f"Kept {len(df)} events from line ({x0}, {y0}) to ({x1}, {y1})")

    df_full = df.copy()

    # Save full events file
    full_file = fm.events_file(
        data_root=args.data_root,
        source=context["source"],
        dataset=context["dataset"],
        stem=base_stem,
        slice_name=output_slice,
        create_parent=True,
        **line_kwarg,
    )
    df_full.to_csv(full_file, index=False)

    # Save ON polarity events
    df_on = df_full[df_full["p"] == 1]
    on_file = fm.events_file(
        data_root=args.data_root,
        source=context["source"],
        dataset=context["dataset"],
        stem=base_stem,
        slice_name=output_slice,
        polarity="ON",
        create_parent=True,
        **line_kwarg,
    )
    df_on.to_csv(on_file, index=False)

    # Save OFF polarity events
    df_off = df_full[df_full["p"] == 0]
    off_file = fm.events_file(
        data_root=args.data_root,
        source=context["source"],
        dataset=context["dataset"],
        stem=base_stem,
        slice_name=output_slice,
        polarity="OFF",
        create_parent=True,
        **line_kwarg,
    )
    df_off.to_csv(off_file, index=False)

    # Generate info file if line filtering was used
    if args.keep_line is not None:
        kept_event_pixels = len({
            (int(x), int(y))
            for x, y in zip(df_full["x"], df_full["y"])
        })

        info_file = full_file.with_name(full_file.stem.removesuffix("_line" + str(line_label)) + f"_line{line_label}_info.txt")
        image_file = fm.picture_file(
            "line_mask",
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            stem=base_stem,
            slice_name=output_slice,
            line=line_label,
            create_parent=True,
        )
        save_line_info(
            info_file,
            line_label=line_label,
            keep_line=args.keep_line,
            line_radius=args.line_radius,
            width=args.width,
            height=args.height,
            input_filename=str(input_file),
            output_filenames={
                "all": str(full_file),
                "ON": str(on_file),
                "OFF": str(off_file),
            },
            line_mask_filename=str(image_file),
            total_events=total_events,
            kept_events=len(df_full),
            kept_event_pixels=kept_event_pixels,
        )
        save_line_mask_image(
            image_file,
            kept_pixels,
            args.width,
            args.height,
            args.keep_line,
            args.line_radius,
            show=args.show_line_mask,
        )
