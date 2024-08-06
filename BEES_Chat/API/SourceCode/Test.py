from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI

import os

load_dotenv(find_dotenv())

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('Azure_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('Azure_OPENAI_API_BASE')
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-09-15-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "qnagpt5"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "bradsol-embeddings"


llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"), temperature=0, max_tokens=2000
)

llm.generate(messages=["Shameerpet bus timing"])