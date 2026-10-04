from src.config import RAW_DATA

# search recursively for files and return the short name to path mapping
def find_files(root_path, delimiter="."):
    """
    Search recursively for files in the given root path and return a dictionary mapping the short name of each file to its full path.

    """
    files = {}
    for file in root_path.rglob("*"):
        if not file.is_file():
            continue
        if not (file.name.endswith(".csv") or file.name.endswith(".csv.gz")):
            continue
        short_name = file.name[:file.name.find(delimiter)]
        
        if short_name in files:
            raise ValueError(f"Duplicate file name {short_name} found in {file} and {files[short_name]}.")
        
        files[short_name] =  file
        
    return files
