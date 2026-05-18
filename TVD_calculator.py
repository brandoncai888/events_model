import pandas as pd

import argparse

import file_manager as fm


def calculate_tvd(df):
    ## assume each row has equal weight
    total_rows = len(df)
    if total_rows == 0:
        print("DataFrame is empty. Cannot calculate TVD.")
        return None
    
    data_total = df['data'].sum()
    expected_total = df['expected'].sum()   
    df['data_norm'] = df['data'] / data_total
    df['expected_norm'] = df['expected'] / expected_total
    tvd = 0 
    for row in df.itertuples():
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
        df = pd.read_csv(filename)
        tvd = calculate_tvd(df)
        tvd_values.append(tvd)
        max_file_len = max(max_file_len, len(filename))
    with open(args.output, 'w') as f:
        f.write("TVD Calculation Results\n")
        for filename, tvd in zip(filenames, tvd_values):
            f.write(f"{filename:<{max_file_len}} |  {tvd}\n")

if __name__ == "__main__":
    main()
