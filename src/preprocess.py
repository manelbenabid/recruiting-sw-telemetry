from src.utils import standstill_stats
from src.loader import load

OFFSET_CHANNELS = {
    "imu_angular_rate": ["x", "y", "z"],
    "imu_acceleration": ["x"],
}


def remove_offsets(channels=OFFSET_CHANNELS):
    """subtract standstill offsets. 
    
    returns dict {file: corrected DataFrame} and the offsets used
    
    """
    stats = standstill_stats(sensors=channels)
    stats = stats.pivot_table(index=["file", "column"], columns="window", values="mean")
    offsets = stats.mean(axis=1)

    corrected = {}
    for file, measure_values in channels.items():
        df = load(file).drop(columns=["imu_temperature"], errors="ignore")
        for col in measure_values:
            df[col] = df[col] - offsets.loc[(file, col)]
        corrected[file] = df

    return corrected, offsets