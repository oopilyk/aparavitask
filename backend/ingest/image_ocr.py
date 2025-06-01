from PIL import Image
import pytesseract
import os
from .utils import get_file_metadata

def extract_text_from_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return {
            "text": text.strip(),
            "metadata": get_file_metadata(file_path)
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "text": "",
            "metadata": get_file_metadata(file_path)
        }
