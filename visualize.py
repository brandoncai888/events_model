import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.animation as animation
import time
import argparse
import sys

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
        unique_vals = sorted(df[p_col].unique()) # Get from whole df to keep colors consistent
        bright_colors = generate_high_contrast_colors(len(unique_vals))
        
        for i, val in enumerate(unique_vals):
            subset = slice_df[slice_df[p_col] == val]
            if not subset.empty:
                ax.scatter(subset['x'], subset['y'], color=bright_colors[i], s=1, label=f"{p_col}: {val}", alpha=0.9)

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

def animate_event_stream(df, start_t, end_t, fps, width, height, p_col='p', color_events=True, show_legend=True, save_path=None, suppress_show=False):
    print(f"Preparing animation from t={start_t}s to t={end_t}s at {fps} FPS...")
    
    frame_time = 1.0 / fps
    num_frames = int((end_t - start_t) / frame_time)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_vals = sorted(df[p_col].unique())
    num_vals = len(unique_vals)
    
    if not color_events:
        cmap = ListedColormap(['black', 'white'])
        vmin, vmax = 0, 1
        legend_elements = [Patch(facecolor='white', label='Event', edgecolor='gray')]
    else:
        bright_colors = generate_high_contrast_colors(num_vals)
        all_colors = ['black'] + bright_colors
        cmap = ListedColormap(all_colors)
        vmin, vmax = 0, num_vals
        legend_elements = [Patch(facecolor=bright_colors[i], label=f"{p_col}: {val}", edgecolor='gray') for i, val in enumerate(unique_vals)]
    
    im = ax.imshow(np.zeros((height, width)), cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    # Initialize the title object so we can update and return it later
    title_obj = ax.set_title("") 
    
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
            for i, val in enumerate(unique_vals):
                mapped_val = i + 1 
                subset = frame_df[frame_df[p_col] == val]
                if not subset.empty:
                    hist, _, _ = np.histogram2d(subset['y'], subset['x'], bins=[height, width], range=[[0, height], [0, width]])
                    img_data[hist > 0] = mapped_val 
                    
        im.set_data(img_data)
        
        if t0 > 0:
            # Update progress in-place so long animations do not flood the terminal.
            print(f"\rEvent Stream | t = {t0:.3f}s | Frame Events: {len(frame_df)}", end="", flush=True)
            title_obj.set_text(f"Event Stream | t = {t0:.3f}s\nFrame Events: {len(frame_df)}")
        
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
        ani.save(save_path, writer='ffmpeg', fps=fps)
        print()
        print("Video saved successfully.")
        
    print("Click on the animation window to pause/resume.")
    if not suppress_show:
        plt.show()

if __name__ == "__main__":
    suppress_show = len(sys.argv) > 1

    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--folder", type=str, default="data", help="Base folder to save results (default: current directory).")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate
    SUFFIX = f"{LAMBDA_RATE}Hz_{SIM_DURATION}s"

    # Load the event data from the CSV file generated in the previous step
    filename = f"{args.folder}/poisson_noise_{SUFFIX}.csv"
    event_data = pd.read_csv(filename)
    
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
        start_t=0.0,
        end_t=SIM_DURATION, 
        fps=30,
        width=SENSOR_WIDTH, 
        height=SENSOR_HEIGHT,
        p_col='p',           
        color_events=True,
        show_legend=False,
        save_path = f"{args.folder}/poisson_noise_{SUFFIX}_animation.mp4"
    )
