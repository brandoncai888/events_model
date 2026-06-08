import pandas as pd

# Define file paths
input_txt_path = "scene0.txt"  # Update this to your actual file path
output_csv_path = "scene0.csv"

print("Reading events.txt into Pandas...")

# 1. Read the space-separated text file efficiently
# 'sep=r"\s+"' handles one or more spaces as delimiters
df = pd.read_csv(
    input_txt_path, sep=r"\s+", header=None, names=["t", "x", "y", "p"]
)

# 2. Optional: Fix Samsung Polarity inversion if needed
# If the original dataset uses 0 and 1, this flips them (0 becomes 1, 1 becomes 0)
# If it uses -1 and 1, use: df['p'] = df['p'] * -1
# Uncomment the line below if your pipeline requires standard polarity:
df['p'] = 1 - df['p']

df['t'] = df['t'] - min(df['t'])


print("Preview of the DataFrame:")
print(df.head())

print(f"x: {df['x'].min()} to {df['x'].max()}")
print(f"y: {df['y'].min()} to {df['y'].max()}")
print(f"t: {df['t'].min()} to {df['t'].max()}")

print(f"\nSaving to CSV format at: {output_csv_path}...")

# 3. Export to CSV without the index column to keep it clean
df.to_csv(output_csv_path, index=False)

print("Done! CSV successfully created.")