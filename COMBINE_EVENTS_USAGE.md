# Event Combination Script

## Overview
`combine_events.py` combines multiple event sets, normalizing each one to start at time 0, then merging them into a single sorted CSV file.

## Features
- Supports combining 2, 3, or more event sets
- Works with direct file paths or managed dataset references
- Automatically aligns min time of each set to 0
- Combines and sorts all events by time
- Auto-generates descriptive output filenames or accepts custom ones

## Usage

### Basic Examples

#### Combine two files directly:
```bash
python combine_events.py \
  --sources data_equal_counts/poisson_noise_0.1Hz_400.0s.csv \
            data_equal_counts/poisson_noise_10.0Hz_4.0s.csv \
  --datasets - - \
  --slices - -
```

#### Combine three files with custom output:
```bash
python combine_events.py \
  --sources data_equal_counts/poisson_noise_0.1Hz_400.0s.csv \
            data_equal_counts/poisson_noise_0.2Hz_200.0s.csv \
            data_equal_counts/poisson_noise_10.0Hz_4.0s.csv \
  --datasets - - - \
  --slices - - - \
  --output my_combined_events.csv
```

#### Using managed dataset references (if using data/ folder structure):
```bash
python combine_events.py \
  --sources noise object egomotion \
  --datasets 1.0Hz 45 1 \
  --slices - 2.67_2.71 1.02_1.08
```

## Command-line Arguments

### Required Arguments:
- `--sources`: List of source names or file paths
  - Can be: managed dataset names (e.g., `noise`, `object`) 
  - Or: direct file paths (e.g., `data_equal_counts/poisson_noise_0.1Hz_400.0s.csv`)

- `--datasets`: List of dataset names or `-` when using file paths
  - For managed: dataset identifier (e.g., `1.0Hz`, `45`, `1`)
  - For files: use `-`

- `--slices`: List of slice names or `-` for no slice
  - For managed: slice identifier (e.g., `2.67_2.71`)
  - For files: use `-`

### Optional Arguments:
- `--data_root`: Root folder for managed data files (default: `data`)
- `--polarity`: Filter for object event files (choices: `ON`, `OFF`)
- `--output`: Custom output CSV filename (default: auto-generated)

## Output

### Output File Format:
- CSV file with columns: `x`, `y`, `t`, `p`
- All events sorted by time `t` in ascending order
- Each input set normalized so its minimum time becomes 0

### Filename Convention:
- Auto-generated: `combined_events_{set1}+{set2}+...csv`
- Example: `combined_events_poisson_noise_0.1Hz_400.0s+poisson_noise_10.0Hz_4.0s.csv`

## Example Output

Given:
- File 1: 3,596,448 events, original time range: 0 to 399.999661s → normalized to 0 to 399.999661s
- File 2: 3,599,074 events, original time range: 0 to 3.999999s → normalized to 0 to 3.999999s

Result:
- Combined: 7,195,522 events, time range: 0 to 399.999661s (all events merged and sorted)

## Implementation Notes

- Each event set's time is independently normalized by subtracting its minimum time value
- This means if you have two recordings of 5 seconds each, both starting at time 0:
  - Set 1: times 0→5s becomes 0→5s (no change since min=0)
  - Set 2: times 0→5s becomes 0→5s (no change since min=0)
  - Combined: 10,000+ events with times 0→5s (mixed from both sets)

- If event sets have different starting times (e.g., one from 1.02s to 1.08s):
  - Original times 1.02→1.08s become 0→0.06s after normalization
  - Then combined with other sets

- Events are sorted chronologically by their normalized times
