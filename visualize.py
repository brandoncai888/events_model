import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.animation as animation
import time
import argparse
from pathlib import Path

import file_manager as fm

def generate_high_contrast_colors(num_colors):
    """
    Generates an arbitrary number of bright, distinct colors.
    By using the HSV color space and forcing high Saturation (S) 
    and high Value/Brightness (V), we guarantee contrast against black.
    """
    # Evenly space the hues across the color wheel (0 to 1)
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    
    # S=0.85 (vivid, not washed out), V=0.95 (very bright)
    colors = [mcolors.hsv_to_rgb([h, 0.85, 0.95]) for h in hues]
    return colors

def visualize_single_slice_scatter(df, start_t, frame_duration, width, height, p_col='p', color_events=True, show_legend=True, save_path=None, suppress_show=False):
    end_t = start_t + frame_duration
    print(f"Rendering scatter plot from t={start_t}s to t={end_t}s...")
    
    mask = (df['t'] >= start_t) & (df['t'] < end_t)
    slice_df = df[mask]
    
    if slice_df.empty:
        print("No events found in this time window.")
        return

    # Use facecolor='black' on the figure itself to match the axes if saving
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111)
    
    if not color_events:
        ax.scatter(slice_df['x'], slice_df['y'], c='white', s=1, alpha=0.9, label='Event')
    else:
        colors = generate_high_contrast_colors(2)
        for polarity in [0, 1]:
            subset = slice_df[slice_df[p_col] == polarity]
            if not subset.empty:
                ax.scatter(subset['x'], subset['y'], color=colors[polarity], s=1, label=f"{p_col}: {polarity}", alpha=0.9)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('black')
    
    # Text colors need to be white to be visible if the figure background is black
    ax.set_title(f"Event Scatter | t = {start_t}s to {end_t}s\nEvents: {len(slice_df):,}", color='white')
    ax.set_xlabel("X coordinate", color='white')
    ax.set_ylabel("Y coordinate", color='white')
    ax.tick_params(colors='white')
    
    if show_legend:
        # Create legend with a dark background to blend in
        legend = ax.legend(markerscale=3, loc='upper left', bbox_to_anchor=(1.05, 1), facecolor='black', edgecolor='gray')
        for text in legend.get_texts():
            text.set_color("white")
        
    plt.tight_layout()
    
    # Save the image if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Image saved successfully to {save_path}")
        
    if not suppress_show:
        plt.show()

def animate_event_stream(df, start_t, end_t, fps, width, height, p_col='p', color_events=True, show_legend=True, save_path=None, suppress_show=False, scale_factor=1.0):
    print(f"Preparing animation from t={start_t}s to t={end_t}s at {fps} FPS...")

    required_cols = {'x', 'y', 't'}
    if color_events:
        required_cols.add(p_col)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_cols)}")

    in_bounds = (df['x'] >= 0) & (df['x'] < width) & (df['y'] >= 0) & (df['y'] < height)
    if not in_bounds.all():
        dropped = len(df) - int(in_bounds.sum())
        print(
            f"Warning: dropping {dropped:,}/{len(df):,} events outside "
            f"the {width}x{height} frame. Check the AEDAT address decoder "
            "or pass the correct --width/--height."
        )
        df = df[in_bounds].copy()

    if df.empty:
        raise ValueError("No in-bounds events to animate.")
    
    frame_time = 1.0 / fps
    num_frames = max(1, int(np.ceil((end_t - start_t) / frame_time)))
    size = (8,6)
    if scale_factor != 1.0:
        size = (12,9)
    fig, ax = plt.subplots(figsize=size)
    
    if color_events:
        colors = generate_high_contrast_colors(2)
        unique_vals = [val for val in [0, 1] if val in df[p_col].unique()]
        cmap = ListedColormap(['black', colors[0], colors[1]])
        vmin, vmax = 0, 2
        legend_elements = [Patch(facecolor=colors[val], label=f"{p_col}: {val}", edgecolor='gray') for val in unique_vals]
    else:
        unique_vals = []
        cmap = ListedColormap(['black', 'white'])
        vmin, vmax = 0, 1
        legend_elements = [Patch(facecolor='white', label='Event', edgecolor='gray')]
    
    im = ax.imshow(np.zeros((height, width)), cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    # Initialize the title object so we can update and return it later
    title_obj = ax.set_title("\n") 
    
    if show_legend:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    def update(frame_idx):
        t0 = start_t + frame_idx * frame_time
        t1 = t0 + frame_time
        
        mask = (df['t'] >= t0) & (df['t'] < t1)
        frame_df = df[mask]
        
        img_data = np.zeros((height, width))
        
        if not color_events:
            hist, _, _ = np.histogram2d(frame_df['y'], frame_df['x'], bins=[height, width], range=[[0, height], [0, width]])
            img_data[hist > 0] = 1
        else:
            for val in unique_vals:
                subset = frame_df[frame_df[p_col] == val]
                if not subset.empty:
                    hist, _, _ = np.histogram2d(subset['y'], subset['x'], bins=[height, width], range=[[0, height], [0, width]])
                    img_data[hist > 0] = val + 1
                    
        im.set_data(img_data)
        
        if t0 > start_t:
            # Update progress in-place so long animations do not flood the terminal.
            print(f"\rEvent Stream | t = {t0:.4f}s | Frame Events: {len(frame_df)}", end="", flush=True)
            title_obj.set_text(f"Event Stream | t = {t0:.4f}s\nFrame Events: {len(frame_df)}")
        
        # Return BOTH the image and the title object so blit=True redraws them
        return [im, title_obj]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=True)
    
    is_paused = False
    
    def toggle_pause(event):
        nonlocal is_paused
        if is_paused:
            ani.resume()
        else:
            ani.pause()
        is_paused = not is_paused

    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving video to {save_path} (this may take a minute depending on duration)...")
        ani.save(save_path, writer='ffmpeg', fps=fps/scale_factor)
        print()
        print("Video saved successfully.")
        
    print("Click on the animation window to pause/resume.")
    if not suppress_show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=None, help="Data source folder. Defaults to noise unless --filename is a managed object path.")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name. Defaults to '<rate>Hz' for noise.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None, help="Optional polarity suffix for object event files.")
    parser.add_argument("--no_show", action="store_true", help="Suppress plt.show() to avoid opening windows during batch runs.")
    parser.add_argument("--video", type=float, default=float('inf'), help="Duration in seconds for the generated video (default: inf = full duration).")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename prefix (without extension) for the generated CSV and video. If not provided, it will be auto-generated based on rate and duration.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the generated video (default: 30).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds for visualization (default: 0).")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Slowdown factor for the generated video (default: 1.0 = no slowdown).")
    parser.add_argument("--line", type=int, default=None, help="Use line-filtered events file with the specified line number (e.g., --line 1 for line1.csv).")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate

    # Load the event data from the CSV file generated in the previous step
    source = args.source or fm.SOURCE_NOISE
    filename = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=source,
        dataset=args.dataset,
        rate=LAMBDA_RATE,
        duration=SIM_DURATION,
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
    input_slice_name = context["slice_name"]
    event_stem = Path(filename).stem
    event_data = pd.read_csv(filename)
    
    end_t = args.start + min(args.video, SIM_DURATION)
    output_slice_name = input_slice_name
    if output_slice_name is None and (args.start != 0.0 or end_t < SIM_DURATION):
        output_slice_name = fm.time_slice_name(args.start, end_t)
    video_name = "animation"
    if args.line is not None and f"_line{args.line}" not in event_stem:
        video_name += f"_line{args.line}"
    if args.slowdown != 1.0:
        video_name += f"_{round(1 / args.slowdown, 2)}x"
    out_filename = fm.video_file(
        video_name,
        data_root=args.data_root,
        source=source,
        dataset=dataset,
        rate=LAMBDA_RATE,
        duration=SIM_DURATION,
        stem=event_stem,
        slice_name=output_slice_name,
        create_parent=True,
        line=args.line,
    )
    # Visualize using the dynamic color generator, but turning the legend OFF
    # visualize_single_slice_scatter(
    #     df=event_data, 
    #     start_t=1.0, 
    #     frame_duration=0.1, 
    #     width=SENSOR_WIDTH, 
    #     height=SENSOR_HEIGHT,
    #     p_col='p',           
    #     color_events=True,
    #     show_legend=True,
    #     suppress_show=suppress_show
    # ) 
    animate_event_stream(
        df=event_data, 
        start_t=args.start,
        end_t=end_t, 
        fps=args.fps,
        scale_factor=args.slowdown,
        width=SENSOR_WIDTH, 
        height=SENSOR_HEIGHT,
        p_col='p',           
        color_events=True,
        show_legend=False,
        save_path = str(out_filename),
        suppress_show=args.no_show
    )
