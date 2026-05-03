import numpy as np
import pandas as pd
from noise.visualize import *
import time
import argparse
import sys

def generate_poisson_noise(width, height, duration, lambda_hz):
    print(f"Generating noise (Lambda: {lambda_hz} Hz, Duration: {duration}s)...")
    start_time = time.time()
    
    total_pixels = width * height
    avg_total_events = int(total_pixels * duration * lambda_hz)
    n_events = np.random.poisson(avg_total_events)

    t = np.sort(np.random.uniform(0, duration, n_events))
    x = np.random.randint(0, width, n_events)
    y = np.random.randint(0, height, n_events)
    p = np.random.choice([0, 1], n_events)

    df = pd.DataFrame({'x': x, 'y': y, 't': t, 'p': p})
    
    print(f"Generated {n_events:,} events in {time.time() - start_time:.2f} seconds.")
    return df


if __name__ == "__main__":
    suppress_show = len(sys.argv) > 1

    parser = argparse.ArgumentParser(description="Generate Poisson noise event data.")
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson event rate per pixel in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--folder", type=str, default="noise/data", help="Base folder to save results (default: current directory).")
    args = parser.parse_args()

    SENSOR_WIDTH = args.width
    SENSOR_HEIGHT = args.height
    SIM_DURATION = args.duration
    LAMBDA_RATE = args.rate
    SUFFIX = f"{LAMBDA_RATE}Hz_{SIM_DURATION}s"

    # Generating data with 14 arbitrary distinct classes in the 'p' column
    event_data = generate_poisson_noise(SENSOR_WIDTH, SENSOR_HEIGHT, SIM_DURATION, LAMBDA_RATE)
    event_data.to_csv(f"{args.folder}/poisson_noise_{SUFFIX}.csv", index=False)


    animate_event_stream(
        df=event_data, 
        start_t=0.0,
        end_t = SIM_DURATION, 
        fps=30,
        width=SENSOR_WIDTH, 
        height=SENSOR_HEIGHT,
        p_col='p',           
        color_events=True,
        show_legend=False,
        save_path = f"{args.folder}/poisson_noise_{SUFFIX}_animation.mp4",
        suppress_show=suppress_show
    ) 
