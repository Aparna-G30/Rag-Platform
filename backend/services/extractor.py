import fitz  # PyMuPDF
from docx import Document as DocxDocument

def extract_text_from_pdf(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages
            pages.append({"page_number": page_num, "text": text})
    return pages

def extract_text_from_docx(file_path: str) -> list[dict]:
    doc = DocxDocument(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page_number": 1, "text": full_text}]

def extract_text(file_path: str) -> list[dict]:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")