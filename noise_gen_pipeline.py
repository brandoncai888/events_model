import argparse
import subprocess
import sys


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_step(script, rate, duration, width, height, folder):
    command = [
        sys.executable,
        script,
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--width",
        str(width),
        "--height",
        str(height),
        "--folder",
        folder,
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run generate_poisson, inter_event_time, and graphs over rate/duration inputs."
    )
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
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Pair rates and durations by index instead of running every combination.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=".",
        help="Base folder to save results (default: 'results').",
    )
    args = parser.parse_args()

    if args.paired:
        if len(args.rates) != len(args.durations):
            raise ValueError("--paired requires the same number of rates and durations.")
        runs = zip(args.rates, args.durations)
    else:
        runs = ((rate, duration) for rate in args.rates for duration in args.durations)

    for rate, duration in runs:
        print(f"\n=== Pipeline: rate={rate} Hz, duration={duration}s ===")
        run_step("generate_poisson.py", rate, duration, args.width, args.height, args.folder)
        run_step("inter_event_time.py", rate, duration, args.width, args.height, args.folder)
        run_step("graphs.py", rate, duration, args.width, args.height, args.folder)


if __name__ == "__main__":
    main()
