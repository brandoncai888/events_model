import numpy as np
import pandas as pd
from visualize import *
import time

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
    df['t'] = df['t'].round(6)
    
    print(f"Generated {n_events:,} events in {time.time() - start_time:.2f} seconds.")
    return df


if __name__ == "__main__":
    SENSOR_WIDTH = 346
    SENSOR_HEIGHT = 260
    SIM_DURATION = 5.0      
    LAMBDA_RATE = 5.0       
    
    # Generating data with 14 arbitrary distinct classes in the 'p' column
    event_data = generate_poisson_noise(SENSOR_WIDTH, SENSOR_HEIGHT, SIM_DURATION, LAMBDA_RATE)
    event_data.to_csv(f"poisson_noise_{LAMBDA_RATE}.csv", index=False)

    # Visualize using the dynamic color generator, but turning the legend OFF
    # visualize_single_slice_scatter(
    #     df=event_data, 
    #     start_t=1.0, 
    #     frame_duration=0.1, 
    #     width=SENSOR_WIDTH, 
    #     height=SENSOR_HEIGHT,
    #     p_col='p',           
    #     color_events=True,
    #     show_legend=True
    # ) 
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
        save_path = f"poisson_noise_{LAMBDA_RATE}_animation.mp4"
    ) 