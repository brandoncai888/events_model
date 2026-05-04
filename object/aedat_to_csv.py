import dv
import pandas as pd

import argparse

import numpy as np
import pandas as pd

def aedat2_to_csv(input_path, output_path, camera='davis'):
    print(f"Opening {input_path} (AEDAT 2.0)...")
    
    with open(input_path, 'rb') as f:
        # 1. Skip the text header (lines starting with '#')
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                break
        
        # Step back one line because we accidentally read the first byte of binary data
        f.seek(-len(line), 1)
        
        # 2. Read the rest of the binary file
        raw_data = f.read()
        
    # 3. AEDAT 2.0 consists of alternating 32-bit integers: [Address, Timestamp, Address, Timestamp...]
    # Read as big-endian 32-bit unsigned integers ('>u4')
    data = np.frombuffer(raw_data, dtype='>u4')
    
    # Split the array into addresses (even indices) and timestamps (odd indices)
    addresses = data[0::2]
    timestamps = data[1::2]
    
    # 4. Decode the addresses using bitwise shifts
    if camera.lower() == 'davis':
        # Default for DAVIS cameras (e.g., DAVIS240, DAVIS346)
        x_coords = (addresses >> 17) & 0x000003FF
        y_coords = (addresses >> 2) & 0x00007FFF
        polarities = (addresses >> 1) & 0x00000001
    elif camera.lower() == 'dvs128':
        # Default for older DVS128 cameras
        x_coords = (addresses >> 1) & 0x0000007F
        y_coords = (addresses >> 8) & 0x0000007F
        polarities = addresses & 0x00000001
    else:
        raise ValueError("Unsupported camera type. Choose 'davis' or 'dvs128'.")

    # 5. Package into a DataFrame and save
    df = pd.DataFrame({
        't': (timestamps - min(timestamps)) / 1e6,  # Convert from microseconds to seconds
        'x': x_coords,
        'y': y_coords,
        'p': polarities
    })

    df.to_csv(output_path, index=False)
    print(f"Success! Exported {len(df)} events to {output_path}")

def parse_str_list(value):
    return [str(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run generate_poisson, inter_event_time, and graphs over rate/duration inputs.")
    parser.add_argument(
        "--aedat",
        type=parse_str_list,
        required=True,
        help="Comma-separated .aedat files: events1,events2",
    )
    parser.add_argument("--folder", type=str, default="data", help="Base folder to find .aedat files and save CSVs")
    args = parser.parse_args()

    for aedat_file in args.aedat:
        input_file = f"{args.folder}/{aedat_file}.aedat"
        output_file = f"{args.folder}/{aedat_file}.csv"
        print(f"\n\nProcessing {input_file}...")
        aedat2_to_csv(input_file, output_file)
