import pdfplumber
import pytesseract
from PIL import Image
import os

def extract_text_from_file(filepath):
    ext = filepath.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif ext in {'png', 'jpg', 'jpeg', 'heic'}:
        return extract_text_from_image(filepath), 1
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(pdf_path):
    text = ""
    page_count = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text += page.extract_text() or ""  # skip pages with no extractable text
                text += "\n"
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")
    return text.strip(), page_count

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error reading image: {e}")