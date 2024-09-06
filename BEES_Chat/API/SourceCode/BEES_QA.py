import collections
import re
import time
from difflib import SequenceMatcher
import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import langchain_community.callbacks
from azure.cosmos import CosmosClient, PartitionKey
from .azure_no_sql import (AzureCosmosDBNoSqlVectorSearch, )
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .Log import Logger
import datetime

from dotenv import load_dotenv

load_dotenv(find_dotenv())
logger = Logger()
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def text_similarity(a, b):
    vectorizer = TfidfVectorizer().fit_transform([a, b])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]


# Load environment variables from .env file
load_dotenv()
cosmos_key = os.getenv('WebChat_Key')
cosmos_database = os.getenv('WebChat_DB')
cosmos_collection = os.getenv('WebChatChunk_Container')
cosmos_vector_property = "embedding"
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('Azure_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('Azure_OPENAI_API_BASE')
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv('Azure_OpenAIVersion')
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = os.getenv('Azure_OpenAIDeploymentName')
os.environ["AZURE_EMBEDDINGS_DEPLOYMENT_NAME"] = os.getenv('Azure_EmbeddingDeploymentName')

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

openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv('Azure_OPENAI_API_KEY'),
)

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
    search_kwargs={"k": 7},
    return_source_documents=True,
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"), temperature=0.5, max_tokens=500
)

prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system",
     """
Given the conversation history and the user's current question:
Determine the Relevance of the Current Question:
If the current question is related to previous questions or topics, slightly rephrase it while keeping its original meaning to fit the context. Generate a search query that incorporates this adjusted question and relevant parts of the conversation history.
If the current question is unrelated to any prior conversation, ignore the previous chat history and generate a search query solely based on the rephrased version of the current question. The rephrasing should be minimal and retain the question's original intent to optimize it for querying into a vector store.
Generate and Provide the Search Query:
Always output a search query that is appropriate for finding relevant information, even if the exact answer is not known.
If the same question has been asked before, return the previously generated query without any changes.
Output Format:
Ensure the output is a single, clear search query text based on the steps above.
""")
])

history_aware_retriever = langchain.chains.history_aware_retriever.create_history_aware_retriever(
    llm, qa_retriever, prompt_search_query)

### Answer question ###
system_prompt = ("""
You are a highly knowledgeable and concise assistant specializing in Beep chatbot question-answering tasks. Please follow these guidelines:

1. Answer only with relevant information derived from the provided context.
2. Provide precise and concise answers within the context.
3. Ensure your answers are grammatically correct and complete sentences.
4. If the context does not contain the exact answer, use related information and synonyms to provide the most relevant answer possible. If no related information is found, state "Sorry, I don't have information."
5. Do not assume or infer information that is not explicitly mentioned in the context.
6. Do not include personal opinions or interpretations.
7. Avoid redundant information; be direct and to the point.
8. Prioritize clarity and relevance in your answers.
9. Cite specific parts of the context when forming your answer.
10. Avoid using ambiguous language; be as specific as possible.
11. If there are multiple relevant pieces of information in the context, integrate them into a cohesive answer.
12. If a question is ambiguous, state the ambiguity and request clarification.
13. Do not provide general knowledge or background information unless explicitly requested.
14. If the answer requires multiple parts, number each part clearly.
15. If a question relates to the following categories, provide the appropriate response with Beep in it:
    - greeting
    - general inquiry
    - conversation ender
    - thank you
16. Avoid general knowledge questions, and state "Sorry, I don't have information."
17. If the question is related to bus schedules or routes, provide the answer in the following HTML table format:
<table>
<tr>
<th>Area</th>
<th>Time</th>
</tr>
</table>
18. If the question is related to a route number or bus number, fetch the route number in context and provide the answer.
19. Remove the phrase "context states" from the answer.
20. Provide relevant answers for synonyms found in the context.
21. If the question relates to any upcoming, next, or previous holidays, including specific months, weeks, dates, days, or long weekends, refer only to the list of holidays provided in the context. Use the current calendar date to accurately determine and provide information based on the context.
has context menu
22. When asked for updates, news, or information on specific topics, search the context provided for any direct mentions of these topics or related details. Provide a precise answer using the exact information found in the context, and do not consider the current date or any previous conversation history. Always treat each question independently, ensuring that the response is based solely on the provided context. If the context contains relevant information about the topic, summarize that directly.
23. Single-word Questions:
If the question contains only a single word, identify the word in the context and provide related information or a category it belongs to. For example, if "Pune" is mentioned in a list or category (like cities or regions), provide the information or context that categorizes it. If no related context is available, state "Sorry, I don't have information."
24. For every answer generated, mention "Please check the source link for more details." This ensures users are directed to the original source for comprehensive information.
**Stay on topic:** Answer the question based solely on the information in the context. Do not use any outside knowledge.

Context: {context}
Current Date:""" + str(datetime.date.today())
                 )

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
    # for content in ["does not provide", "not found", "does not contain", "not provided", "does not mention",
    #                 "does not", "don't have"]:
    #     if content.lower() in answer.lower():
    #         return "Sorry, I don't have information. Could you please provide more precise question", ''
    if "BeepChat Assistant" in answer or "unable to" in answer or "feel free" in answer or "to ask" in answer or "How can I help you" in answer or "assist you" in answer:
        return answer, ''
    return answer, link


def handle_greet(human):
    messages = [
        (
            "system",
            "You are a highly knowledgeable and concise assistant specializing in Beep chatbot greetings handler: " \
            "Provide response for the query."
            "If question is not related to greeting, general queries, thanks or conversation ends State 'Sorry, I don't have information.'"
            "Avoid questions other than above categories, State 'Sorry, I don't have information.'"
            "Along with greetings if any questions is asked, then State 'Sorry, I don't have information.'",
        ),
        ("human", human),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content


def contains_greeting(text):
    # List of common greeting words
    greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']

    # Convert the input text to lowercase for case-insensitive comparison
    text_lower = text.lower()

    # Check if any greeting word is in the text
    return any(greeting in text_lower for greeting in greetings)


def AzureCosmosQA(human, session_id):
    try:
        start_time = time.time()
        human = human.lower()
        with langchain_community.callbacks.get_openai_callback() as cb:
            if contains_greeting(human):
                result = handle_greet(human)
                if "don't have" not in result:
                    return result, cb.total_tokens, cb.total_cost, ''
            response = QA_chain.invoke(
                {"input": human},
                config={
                    "configurable": {"session_id": session_id},
                },
            )
            source_links = [doc.metadata['source'] for doc in response["context"] if 'source' in doc.metadata]
            context = [doc.page_content for doc in response["context"]]
            response = response["answer"]
            print(response)
            similarity = text_similarity(str(context), response)
            print(similarity)
            print("\n\n\n")
            if source_links:
                source_link = source_links[0]
            else:
                source_link = ''
                # response = "Sorry, I don't have information. Could you please provide more precise question"
            if "<table>" not in response:
                pass
                # if similarity < 0.030:
                #   print("score mismatched", similarity)
                 #   source_link = ''
                   # response = "Sorry, I don't have information. Could you please provide more precise question"
            #source_link = re.sub(r'.*Files', '', source_link)
            source_link = source_link.replace("D:\\Webapplication\\BEEP\\", "")
            response, source_link = post_process_answer(str(context), response, source_link)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            end_time = time.time()
            # Calculate the time difference
            elapsed_time = end_time - start_time
            print(f"Time elapsed between lines: {elapsed_time} seconds")
            return response, cb.total_tokens, cb.total_cost, source_link
    except Exception as e:
        error_details = logger.log(f"Error occurred in fetching response:{str(e)}", "Error")
        raise Exception(error_details)
