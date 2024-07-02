import collections
import re

import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import langchain_community.callbacks
from azure.cosmos import CosmosClient, PartitionKey
from .azure_no_sql import (AzureCosmosDBNoSqlVectorSearch, )
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from .Log import Logger

from dotenv import load_dotenv

load_dotenv(find_dotenv())
logger = Logger()
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Load environment variables from .env file
load_dotenv()
cosmos_key = os.getenv('WebChat_Key')
cosmos_database = os.getenv('WebChat_DB')
cosmos_collection = os.getenv('WebChatChunk_Container')
cosmos_vector_property = "embedding"
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('Azure_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('Azure_OPENAI_API_BASE')
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-09-15-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "qnagpt5"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "bradsol-embeddings"

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

cosmos_client = CosmosClient(os.getenv('WebChat_EndPoint'), cosmos_key)
database_name = cosmos_database
container_name = cosmos_collection
partition_key = PartitionKey(path="/id")
cosmos_container_properties = {"partition_key": partition_key}
cosmos_database_properties = {"etag": None, "match_condition": None}

openai_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = AzureCosmosDBNoSqlVectorSearch(embedding=openai_embeddings,
                                             cosmos_client=cosmos_client,
                                             database_name=database_name,
                                             container_name=container_name,
                                             vector_embedding_policy=vector_embedding_policy,
                                             indexing_policy=indexing_policy,
                                             cosmos_container_properties=cosmos_container_properties,
                                             cosmos_database_properties=cosmos_database_properties)
qa_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
    return_source_documents=True,
)

### Contextualize question ###
contextualize_q_system_prompt = ("""Given a chat history and the latest user question, 
reformulate it into a standalone question that can be understood without the context of the chat.
Don't answer the question itself. If the question already makes sense independently, return it as is.
If reformulation is necessary, use concise language and limit the response to three sentences maximum. 
When unsure, state that you don't know."""
                                 )
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"), temperature=0, max_tokens=2000
)
history_aware_retriever = langchain.chains.history_aware_retriever.create_history_aware_retriever(
    llm, qa_retriever, contextualize_q_prompt)

### Answer question ###
system_prompt = ("""
You are a highly knowledgeable and concise assistant specializing in question-answering tasks. Please follow these guidelines:
 
1. Answer only with relevant information derived from the provided context.
2. Provide precise and concise answers strictly within the context.
3. Indicate your confidence in the answer using a scale of 1 (low) to 5 (high).
4. Ensure your answers are grammatically correct and complete sentences.
5. If the context does not contain the answer, state "The answer is not found in the context."
6. Do not assume or infer information that is not explicitly mentioned in the context.
7. Do not include personal opinions or interpretations.
8. Avoid redundant information; be direct and to the point.
9. Prioritize clarity and relevance in your answers.
10. Cite specific parts of the context when forming your answer.
11. Avoid using ambiguous language; be as specific as possible.
12. If there are multiple relevant pieces of information in the context, integrate them into a cohesive answer.
13. If a question is ambiguous, state the ambiguity and request clarification.
 
Context: {context}
""")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = langchain.chains.combine_documents.create_stuff_documents_chain(llm, qa_prompt)

rag_chain = langchain.chains.retrieval.create_retrieval_chain(history_aware_retriever, question_answer_chain)

QA_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    return_source_documents=True
)


def post_process_answer(context, answer, link):
    # Ensure answer is only derived from the context
    for content in ["does not provide", "not found"]:
        if content in answer:
            return "The answer is not available in the provided context.", ''
    return answer, link


def AzureCosmosQA(human, session_id):
    try:
        with langchain_community.callbacks.get_openai_callback() as cb:
            response = QA_chain.invoke(
                {"input": human},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            source_links = [doc.metadata['source'] for doc in response["context"] if 'source' in doc.metadata]
            link_counts = collections.Counter(source_links)
            source_link, most_common_count = link_counts.most_common(1)[0]
            context = [doc.page_content for doc in response["context"]]
            response = response["answer"]
            print(context)
            source_link = re.sub(r'.*Files', '', source_link)
            response, source_link = post_process_answer(str(context), response, source_link)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return response, cb.total_tokens, cb.total_cost, source_link
    except Exception as e:
        error_details = logger.log(f"Error occurred in fetching response:{str(e)}", "Error")
        raise Exception(error_details)
