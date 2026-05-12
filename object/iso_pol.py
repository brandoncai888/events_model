import pandas as pd
from pathlib import Path

import file_manager as fm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze event counts per pixel from a CSV file.")
    parser.add_argument("--filename", type=str, default=None, help="Optional explicit CSV path containing event data with columns 'x', 'y', and 't'.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--source", choices=fm.SOURCES, default=fm.SOURCE_OBJECT, help="Data source folder (default: object).")
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", type=str, default=None, help="Dataset folder name, for example 45Hz.")
    parser.add_argument("--slice", dest="slice_name", type=str, default=None, help="Optional time-slice folder name, for example 2.67_2.71.")
    parser.add_argument("--mintime", type=float, default=0.0, help="Minimum time for filtering events.")
    parser.add_argument("--maxtime", type=float, default=float('inf'), help="Maximum time for filtering events.")
    args = parser.parse_args()

    input_file = fm.find_events_file(
        filename=args.filename,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
    )
    context = fm.context_from_path(
        input_file,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
        slice_name=args.slice_name,
    )
    output_slice = fm.slice_from_window(args.mintime, args.maxtime, slice_name=context["slice_name"])
    base_stem = Path(input_file).stem

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    df = df[df["t"] >= args.mintime]
    df = df[df["t"] < args.maxtime]
    df.to_csv(
        fm.events_file(
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            stem=base_stem,
            slice_name=output_slice,
            create_parent=True,
        ),
        index=False,
    )
    df = df[df["p"] == 1]
    
    df.to_csv(
        fm.events_file(
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            stem=base_stem,
            slice_name=output_slice,
            polarity="ON",
            create_parent=True,
        ),
        index=False,
    )
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    df = df[df["p"] == 0]
    df = df[df["t"] >= args.mintime]
    df = df[df["t"] < args.maxtime]
    df.to_csv(
        fm.events_file(
            data_root=args.data_root,
            source=context["source"],
            dataset=context["dataset"],
            stem=base_stem,
            slice_name=output_slice,
            polarity="OFF",
            create_parent=True,
        ),
        index=False,
    )
