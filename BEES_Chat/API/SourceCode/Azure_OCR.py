from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

endpoint = os.getenv('ocr_endpoint')
key = os.getenv('ocr_key')


def analyze_read(pdf_path):
    # sample document
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    with open(pdf_path, "rb") as pdf:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document", pdf)
        result = poller.result()

    extracted_text = result.content
    # print("Document contains content: ", extracted_text)
    return extracted_text


