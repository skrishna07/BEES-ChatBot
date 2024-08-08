import fitz  # PyMuPDF
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
import os
from SourceCode.Log import Logger
from SourceCode.Azure_OCR import analyze_read

logger = Logger()


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


def image_to_text(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
        return text.strip()
    except Exception as e:
        error_details = logger.log(f"Exception occurred while converting image to text: {e}", "Error")
        raise Exception(error_details)


def pdf_page_to_text(pdf_document, pdf_reader, page_num):
    try:
        text = ""
        pdf_page = pdf_reader.pages[page_num]
        text = pdf_page.extract_text()

        if not text:  # if the text extraction fails, try OCR
            pdf_page = pdf_document.load_page(page_num)
            for img_index, image in enumerate(pdf_page.get_images(full=True)):
                try:
                    xref = image[0]
                    base_image = pdf_document.extract_image(xref)
                    image_data = base_image["image"]
                    temp_image_path = f"temp_image_{page_num}_{img_index}.png"
                    with open(temp_image_path, "wb") as img_file:
                        img_file.write(image_data)
                    text += image_to_text(temp_image_path) + "\n"
                except Exception as e:
                    print(f"Exception occurred while converting image to text: {e}")
                finally:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
        return text.strip()
    except Exception as e:
        error_details = logger.log(f"Exception occurred while converting pdf to text: {e}", "Error")
        raise Exception(error_details)


def process_pdf(file_path, category, id):
    try:
        data = []
        text = analyze_read(file_path)
        text = text.replace('\\n', '')
        text = text.replace('@', '')
        metadata = {'source': file_path, 'category': category, 'unique_id': id}
        data.append(Document(page_content=text, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error occurred while processing PDF: {e}", "Error")
        raise Exception(error_details)


def process_image(file_path, category, id):
    try:
        data = []
        text = analyze_read(file_path)
        text = text.replace('\\n', '')
        text = text.replace('@', '')
        metadata = {'source': file_path, 'category': category, 'unique_id': id}
        data.append(Document(page_content=text, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error occurred while processing image: {e}", "Error")
        raise Exception(error_details)


def process_documents(input_path, category, id):
    try:
        data = []
        if input_path.lower().endswith(".pdf"):
            data.extend(process_pdf(input_path, category, id))
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            data.extend(process_image(input_path, category, id))
        return data
    except Exception as e:
        error_details = logger.log(f"Error occurred while processing attachment doc ocr: {e}", "Error")
        raise Exception(error_details)
