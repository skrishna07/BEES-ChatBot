import langchain_text_splitters
from azure.cosmos import CosmosClient, PartitionKey
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (AzureCosmosDBNoSqlVectorSearch, )
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from .Log import Logger
import os

logger = Logger()
store = {}
load_dotenv(find_dotenv())

cosmos_key = os.getenv('WebChat_Key')
cosmos_database = os.getenv('WebChat_DB')
cosmos_collection = os.getenv('WebChatChunk_Container')
cosmos_vector_property = "embedding"
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('Azure_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('Azure_OPENAI_API_BASE')
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-09-15-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt35"
os.environ["AZURE_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"
os.environ["AZURE_EMBEDDINGS_DEPLOYMENT_NAME"] = "bradsol-ada-embeddings"


indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 384,
        }
    ]
}

database_name = cosmos_database
container_name = cosmos_collection
partition_key = PartitionKey(path="/id")
cosmos_container_properties = {"partition_key": partition_key}
cosmos_database_properties = {"etag": None, "match_condition": None}
cosmos_client = CosmosClient(os.getenv('WebChat_EndPoint'), cosmos_key)
database = cosmos_client.get_database_client(database_name)
container = database.get_container_client(container_name)

openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv('Azure_OPENAI_API_KEY'),
)


def Load_ChunkData(Data):
    try:
        text_splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        docs = text_splitter.split_documents(Data)

        AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=docs,
            embedding=openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties
        )

    except Exception as e:
        error_details = logger.log(f"Error occurred in Loading Chunk Data: {str(e)}", "Error")
        raise Exception(error_details)


def delete_chunk_item(unique_id):
    query = f"SELECT c.id FROM c WHERE c.unique_id = '{unique_id}'"
    try:
        query_items = container.query_items(query, enable_cross_partition_query=True)
        items = list(query_items)
        if items:
            container.delete_item(item=items[0]['id'], partition_key=items[0]['id'])
            logger.log(f"Successfully deleted Chunk Data: {unique_id}", "Info")
    except:
        logger.log(f"No data found for deletion: {unique_id}", "Error")

