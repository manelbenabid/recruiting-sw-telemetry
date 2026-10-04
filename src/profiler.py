"""
this file aims to derive information I can't see because of the huge amount of data.
to flag any missing data, duplicates, invalid data, gaps in recordings, etc.

two files are generated in the log directory, one row per file and one row per signal column
"""
from src.config import GAP_FACTOR
import pandas as pd

def profile_file(df: pd.DataFrame, gap_factor = GAP_FACTOR):
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
    gaps = dt[dt >= gap_factor * T]
    max_gap_s = dt.max()
    n_gaps = len(gaps)
    gap_time_s = (gaps - T).sum() # sum of gaps in the sample
    
    return {
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
    
    

    
    