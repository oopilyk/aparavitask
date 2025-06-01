import os
from datetime import datetime

def get_file_metadata(file_path):
    return {
        "filename": os.path.basename(file_path),
        "filetype": os.path.splitext(file_path)[-1].replace('.', '').lower(),
        "timestamp": datetime.utcnow().isoformat() + "Z"  # UTC in ISO 8601 format
    }

import re
from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    # Split into sentences (basic approach)
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_chunk = []

    total_length = 0
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if total_length + sentence_len > chunk_size:
            chunks.append(" ".join(current_chunk))
            # start new chunk with overlap
            current_chunk = current_chunk[-overlap:] if overlap else []
            total_length = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        total_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
