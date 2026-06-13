import argparse
import pandas as pd
from pathlib import Path
from typing import Union


def slowdown(events: Union[pd.DataFrame, str, Path], factor: float) -> pd.DataFrame:
	"""Return events with their timestamp column `t` multiplied by `factor`.

	Args:
		events: A pandas DataFrame with columns including `t`, or a path to a CSV
			following the universal file format (columns: x,y,t,p,...).
		factor: Multiplier to apply to the `t` column (e.g., 2.0 slows times by 2x).

	Returns:
		A new pandas DataFrame with the adjusted `t` values.

	Notes:
		This function does not write files; if a path is provided it will read
		the CSV and return the modified DataFrame. The original DataFrame
		(or file) is not modified in-place.
	"""
	# Load if a path was provided
	if isinstance(events, (str, Path)):
		df = pd.read_csv(events)
	elif isinstance(events, pd.DataFrame):
		df = events.copy()
	else:
		raise TypeError("events must be a pandas.DataFrame or a path to a CSV file")

	if "t" not in df.columns:
		raise KeyError("Input events must contain a 't' column with timestamps")

	# Ensure numeric
	df["t"] = pd.to_numeric(df["t"], errors="raise") * float(factor)

	return df


def main():
	parser = argparse.ArgumentParser(description="Multiply event timestamps by a slowdown factor.")
	parser.add_argument("input", help="Path to input events CSV file (universal format with columns x,y,t,p)")
	parser.add_argument("--factor", type=float, required=True, help="Slowdown multiplier to apply to timestamps (e.g., 2.0)")
	parser.add_argument("--output", default=None, help="Path to output CSV. If omitted, input file will be overwritten.")

	args = parser.parse_args()
	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	out_path = Path(args.output) if args.output else input_path

	df = slowdown(input_path, args.factor)
	df.to_csv(out_path, index=False)
	print(f"Wrote slowed events to {out_path}")


if __name__ == "__main__":
	main()
