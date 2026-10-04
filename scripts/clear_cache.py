"""
Clear the cache file.
"""

from src.config import CACHE_FILE


if CACHE_FILE.exists():
    CACHE_FILE.unlink()
    print("Deleted cache file: ", CACHE_FILE)
else:
    print("No cache file to delete.")
    
