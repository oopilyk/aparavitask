import os
from datetime import datetime

def get_file_metadata(file_path):
    return {
        "filename": os.path.basename(file_path),
        "filetype": os.path.splitext(file_path)[-1].replace('.', '').lower(),
        "timestamp": datetime.utcnow().isoformat() + "Z"  # UTC in ISO 8601 format
    }
