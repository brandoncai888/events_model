import argparse
import subprocess
import sys

import file_manager as fm


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_step(script, rate, duration, width, height, data_root, video=None):
    command = [
        sys.executable,
        "-m",
        script,
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--width",
        str(width),
        "--height",
        str(height),
        "--data_root",
        data_root,
        "--no_show",
    ]
    if video is not None:
        command.extend(["--video", str(video)])
    print(f"\n\nRunning: {' '.join(command)}\n")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run generate_poisson, inter_event_time, and graphs over rate/duration inputs.")
    parser.add_argument(
        "--rates",
        type=parse_float_list,
        required=True,
        help="Comma-separated Poisson rates in Hz, for example: 0.5,1.0,2.0",
    )
    parser.add_argument(
        "--durations",
        type=parse_float_list,
        required=True,
        help="Comma-separated durations in seconds, for example: 5,10,20",
    )
    parser.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    parser.add_argument("--paired", action="store_true", help="Pair rates and durations by index instead of running every combination.")
    parser.add_argument("--data_root", "--folder", dest="data_root", type=str, default=fm.DEFAULT_DATA_ROOT, help="Root folder for managed data files (default: data).")
    parser.add_argument("--video", type=float, default=0.0, help="Duration in seconds for the generated video (default: 0.0 = no video).")
    args = parser.parse_args()

    if args.paired:
        if len(args.rates) != len(args.durations):
            raise ValueError("--paired requires the same number of rates and durations.")
        runs = zip(args.rates, args.durations)
    else:
        runs = ((rate, duration) for rate in args.rates for duration in args.durations)

    for rate, duration in runs:
        print(f"\n\n\n=== Pipeline: rate={rate} Hz, duration={duration}s ===")
        run_step("noise.generate_poisson", rate, duration, args.width, args.height, args.data_root)
        if args.video > 0:
            run_step("visualize", rate, duration, args.width, args.height, args.data_root, video=args.video)
        run_step("noise.inter_event_time", rate, duration, args.width, args.height, args.data_root)
        run_step("noise.graphs", rate, duration, args.width, args.height, args.data_root)


if __name__ == "__main__":
    main()
