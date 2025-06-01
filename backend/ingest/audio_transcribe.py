import whisper
import os
from .utils import get_file_metadata

model = whisper.load_model("base")

def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        result = model.transcribe(file_path)
        return {
            "text": result["text"].strip(),
            "metadata": get_file_metadata(file_path)
        }
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return {
            "text": "",
            "metadata": get_file_metadata(file_path)
        }
