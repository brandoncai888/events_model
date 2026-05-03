import pandas as pd


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

def main():
    
    tvd_values = []
    max_file_len = 0
    with open('TVD_filenames.txt', 'r') as f:
        filenames = f.readlines()
    for filename in filenames:
        df = pd.read_csv(filename[:-1]) # Remove newline character
        tvd = calculate_tvd(df)
        tvd_values.append(tvd)
        max_file_len = max(max_file_len, len(filename))
    with open('TVD_results.txt', 'w') as f:
        f.write("TVD Calculation Results\n")
        for filename, tvd in zip(filenames, tvd_values):
            f.write(f"{filename[:-1]:<{max_file_len}} |  {tvd}\n")

if __name__ == "__main__":
    main()