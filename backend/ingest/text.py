from PyPDF2 import PdfReader
import os
from .utils import get_file_metadata

def extract_text_from_pdf(file_path):
    if not file_path.endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    all_text = ""

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"
        except Exception as e:
            print(f"Failed to read page {page_num + 1}: {e}")

    return {
        "text": all_text.strip(),
        "metadata": get_file_metadata(file_path)
    }
