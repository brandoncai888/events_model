import pandas as pd
from pathlib import Path

def to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df["t"] = df["t"] / 1000000
    return df

if __name__ == "__main__":
    input_path = "data/other/car/events/car.csv"

    df = pd.read_csv(input_path)
    df = to_seconds(df)
    df.to_csv(input_path, index=False)  