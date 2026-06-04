import pandas as pd

import argparse
from pathlib import Path

import file_manager as fm


TVD_COLUMN_PAIRS = (
    ("data", "expected"),
    ("count", "expected_gaussian_count"),
)


def tvd_columns(df):
    for data_column, expected_column in TVD_COLUMN_PAIRS:
        if data_column in df.columns and expected_column in df.columns:
            return data_column, expected_column
    expected = " or ".join(f"{data}/{expected}" for data, expected in TVD_COLUMN_PAIRS)
    available = ", ".join(df.columns)
    raise ValueError(f"CSV must contain one of these column pairs: {expected}. Found: {available}")


def calculate_tvd(df):
    ## assume each row has equal weight
    total_rows = len(df)
    if total_rows == 0:
        print("DataFrame is empty. Cannot calculate TVD.")
        return None

    data_column, expected_column = tvd_columns(df)
    values = df[[data_column, expected_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        print(f"No numeric values found in {data_column}/{expected_column}. Cannot calculate TVD.")
        return None

    data_total = values[data_column].sum()
    expected_total = values[expected_column].sum()
    if data_total == 0 or expected_total == 0:
        print(f"Zero total in {data_column}/{expected_column}. Cannot calculate TVD.")
        return None

    values['data_norm'] = values[data_column] / data_total
    values['expected_norm'] = values[expected_column] / expected_total
    tvd = 0 
    for row in values.itertuples():
        tvd += abs(row.data_norm - row.expected_norm)
    tvd = tvd / 2
    return tvd


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def managed_filenames(args):
    if args.rates is None and args.durations is None:
        return None
    if args.rates is None or args.durations is None:
        raise ValueError("--rates and --durations must be provided together.")
    if args.paired:
        if len(args.rates) != len(args.durations):
            raise ValueError("--paired requires the same number of rates and durations.")
        runs = zip(args.rates, args.durations)
    else:
        runs = ((rate, duration) for rate in args.rates for duration in args.durations)
    return [
        str(
            fm.picture_file(
                args.picture,
                extension=".csv",
                data_root=args.data_root,
                source=args.source,
                dataset=args.dataset,
                rate=rate,
                duration=duration,
            )
        )
        for rate, duration in runs
    ]


def resolve_csv_filename(filename):
    path = Path(filename)
    if path.suffix.lower() == ".csv":
        return path

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        print(f"Using CSV beside plot image: {csv_path}")
        return csv_path

    return path


def main():
    parser = argparse.ArgumentParser(description="Calculate TVD for plotted binned-value CSVs.")
    parser.add_argument("--filenames", type=str, default="TVD_filenames.txt", help="Text file containing CSV paths, one per line.")
    parser.add_argument("--output", type=str, default="TVD_results.txt", help="Output report path.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=fm.SOURCE_NOISE, help="Source for managed paths.")
    parser.add_argument("--dataset", "--set", dest="dataset", type=str, default=None, help="Dataset folder name for managed paths.")
    parser.add_argument("--rates", type=parse_float_list, default=None, help="Comma-separated rates for managed noise graph CSVs.")
    parser.add_argument("--durations", type=parse_float_list, default=None, help="Comma-separated durations for managed noise graph CSVs.")
    parser.add_argument("--paired", action="store_true", help="Pair rates and durations by index for managed paths.")
    parser.add_argument("--picture", type=str, default="iet_hist", help="Managed picture CSV name, for example iet_hist or frequency_pdf.")
    args = parser.parse_args()
    
    tvd_values = []
    max_file_len = 0
    filenames = managed_filenames(args)
    if filenames is None:
        with open(args.filenames, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
    for filename in filenames:
        if not filename:
            tvd_values.append(None)
            continue
        csv_filename = resolve_csv_filename(filename)
        if csv_filename.suffix.lower() != ".csv":
            print(f"Skipping non-CSV file with no sibling CSV: {filename}")
            tvd_values.append(None)
            max_file_len = max(max_file_len, len(filename))
            continue

        try:
            df = pd.read_csv(csv_filename)
            tvd = calculate_tvd(df)
        except ValueError as err:
            print(f"Skipping {csv_filename}: {err}")
            tvd = None
        tvd_values.append(tvd)
        max_file_len = max(max_file_len, len(filename))
    with open(args.output, 'w') as f:
        f.write("TVD Calculation Results\n")
        for filename, tvd in zip(filenames, tvd_values):
            f.write(f"{filename:<{max_file_len}} |  {tvd}\n")

if __name__ == "__main__":
    main()
