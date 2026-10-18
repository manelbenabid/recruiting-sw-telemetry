from src.config import EXCLUDED_LAPS, WARMUP_LAPS



def comparable_laps(laps):
    """
    
    removed the excluded laps, flags warmup lap and sorts the laps from fastest to slowest
    
    """
    ok = laps[~laps["has_pit"] & ~laps["lap"].isin(EXCLUDED_LAPS)]
    ok = ok.sort_values("lap_time").reset_index(drop=True)
    ok["is_warmup"] = ok["lap"].isin(WARMUP_LAPS)
    return ok

def reference_lap(laps):
    """
    
    takes the fastest comparable lap of the whole session as teh reference lap for the comparison
    
    """
    return int(comparable_laps(laps).iloc[0]["lap"])

