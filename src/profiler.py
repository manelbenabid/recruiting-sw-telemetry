"""
this file aims to derive information I can't see because of the huge amount of data.
to flag any missing data, duplicates, invalid data, gaps in recordings, etc.

two files are generated in the log directory, one row per file and one row per signal column
"""
from src.config import GAP_FACTOR, OUT_LOGS
import pandas as pd
from src.session import find_files
from src.loader import load

# finding gaps
def find_gaps(df: pd.DataFrame, gap_factor = GAP_FACTOR):
    # ensure the gap factor is greater than 1
    if not isinstance(gap_factor, (int, float)) or gap_factor <= 1:
        raise ValueError("gap_factor must be a number greater than 1")    
    
    # steps
    dt = df['t_s'].diff().dropna()
    T  = dt[dt>0].median() # median time between samples
    # gaps, i.e. the steps where dt >= gap_factor * T
    gaps = dt[dt >= gap_factor * T]
    duration_s = gaps
    end_s = df.loc[gaps.index, "t_s"]
    start_s = end_s - gaps
    lost_s = gaps - T
     
    data = pd.DataFrame(
        {
            'start_s': start_s,
            'end_s': end_s,
            'duration_s': duration_s,
            'lost_s': lost_s
        }
    ).reset_index(drop=True)
    
    return data

def profile_file(df: pd.DataFrame, file_name: str, gap_factor = GAP_FACTOR):
    """
    profile the timing of one loaded data file.
    
    input: df: dataframe returned by loader.load() with 't_s' and '_timestamp'
    gap_factor: a step between consecutive samples cunts as a gap when it it's at least gap_factor * the median step.
    
    output:
    dict with start, end and duration (s), sample count, whether the timestamps are sorted, number of duplicate timestapms, sampling rate (Hz), number of gaps,
    largest step (s), and total time lost in gaps (s)   
    
    """
    
    
    # ensure the gap factor is greater than 1
    if not isinstance(gap_factor, (int, float)) or gap_factor <= 1:
        raise ValueError("gap_factor must be a number greater than 1")    
    start_s = df['t_s'].min()
    end_s = df['t_s'].max()
    duration_s = end_s - start_s
    n_samples = len(df)
    
    # steps
    dt = df['t_s'].diff().dropna() # the dropna is for the first row
    # are the timestamps sorted?
    is_sorted = df['t_s'].is_monotonic_increasing
    # check for duplicates
    n_duplicates = df["_timestamp"].duplicated().sum()
    # rate
    T  = dt[dt>0].median() # median time between samples
    rate_hz = 1/T if T>0 else float('nan')
    # gaps, i.e. the steps where dt >= gap_factor * T
    gaps = find_gaps(df, gap_factor)
    max_gap_s = dt.max()
    n_gaps = len(gaps)
    gap_time_s = gaps["lost_s"].sum() # sum of gaps in the sample
    
    return {
        "file": file_name,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "n_samples": n_samples,
        "is_sorted": is_sorted,
        "n_duplicates": n_duplicates,
        "rate_hz": rate_hz,
        "n_gaps": n_gaps,
        "max_gap_s": max_gap_s,
        "gap_time_s": gap_time_s,
    }

# investigating column stats
def profile_columns(df: pd.DataFrame, file_name: str):
    
    """
    profile the values of each signal column in a file.
    
    param: df from loader.load() and the short_name
    
    output: 
    list of dicts, one per signal column.
    with the number of missing values, min, max, number of unique values,
    and the longest run of identical consecutive values: its length in
    samples, its duration (s) and its start time (s).
    
    """
    
    
    rows = []
    for col in df.columns:
        if col in ("t_s", "_timestamp"):
            continue
        
        s = df[col]
        # each change of value starts a new run; cumsum gives every run its own id
        run_id = s.ne(s.shift()).cumsum()
        run_sizes = s.groupby(run_id).size()
        longest_id = run_sizes.idxmax()
        run_t = df.loc[run_id == longest_id, "t_s"]

        rows.append({
            "file": file_name,
            "column": col,
            "n_nan": s.isna().sum(),
            "min": s.min(),
            "max": s.max(),
            "n_unique": s.nunique(),
            "longest_run": run_sizes.max(),
            "longest_run_s": run_t.max() - run_t.min(),
            "longest_run_start_s": run_t.min(),
        })

    return rows


## the final boss
def run_profile():
    
    """
    run the profile_file and profile_columns analysis on all files and save them into the logs.
    
    """
    
    
    file_rows = []
    column_rows = []
    files = find_files()
    for file in files.keys():
        df = load(file)
        file_rows.append(profile_file(df, file_name = file))
        column_rows.extend(profile_columns(df, file))
        
    
    file_rows_df = pd.DataFrame(file_rows)
    column_rows_df = pd.DataFrame(column_rows)
    
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    
    #write to output files
    file_rows_df.to_csv(OUT_LOGS / 'profile_files.csv', index=False)
    column_rows_df.to_csv(OUT_LOGS / 'profile_columns.csv', index=False)
    
    
    return file_rows_df,column_rows_df





