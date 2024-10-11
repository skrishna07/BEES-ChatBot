import collections
import re
import time
from difflib import SequenceMatcher
import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import langchain_community.callbacks
from azure.cosmos import CosmosClient, PartitionKey
from .azuresearch import AzureSearch
from .azure_no_sql import (AzureCosmosDBNoSqlVectorSearch, )
from langchain_community.retrievers import AzureAISearchRetriever

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
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
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
vector_store_address = os.getenv('AISearhVectorstoreAddress')
vector_store_password = os.getenv('AIvector_store_password')
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

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv('Azure_OPENAI_API_KEY'),
)

embedding_function = embeddings.embed_query

index_name: str = "cosmosdb-indexnew1"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embedding_function,
    vector_search_dimensions=3072,
    semantic_configuration_name="sematic_1"
)
qa_retriever = vector_store.as_retriever(
    search_type="semantic_hybrid",
    k=1,
    return_source_documents=True,

)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"), temperature=1, max_tokens=500
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
21. When answering questions about holidays—whether upcoming, next, previous, or specific to a month, week, date, day, or long weekend—  adhere strictly to the following rules:
 
Context-Restricted Responses: Only provide answers based on the holidays listed in the provided holiday calendar. Do not include any holidays or information not present in the context.
 
Specific Timeframe Questions: If a user asks about holidays within a specific timeframe (such as a particular month, week, or date range), only mention the holidays that fall within that specified timeframe.
 
No Generic Information: Avoid providing general information about holidays not listed in the provided holiday calendar. Focus solely on the holidays mentioned in the context.
 
Optional Holidays Information: If the user asks about optional holidays, specify that employees can choose any two from the provided optional holidays list, subject to managerial approval.
 
Regional Specificity: Be specific about which holidays are applicable to which locations (e.g., Telangana, Uttarakhand, Himachal Pradesh) as per the provided data. If a user asks about holidays for a specific region, ensure that only holidays relevant to that region are mentioned.
 
Format of Answer: When listing holidays, provide the holiday name, date, day of the week, and the regions where the holiday is applicable.
22. For news-related questions (e.g., updates, latest news,awards or specific details about a topic), search the context for relevant mentions and provide a concise summary of the  pertinent information. If the question is vague or general, offer a brief overview of the main news points. Always use only the information provided in the context, and if no relevant information is found, state "Sorry, I don't have information."
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


def handle_source(human):
    messages = [
        (
            "system",
            "You are a highly knowledgeable and concise assistant specializing in Beep chatbot greetings handler: " \
            "Provide response for the query."
            """If the question pertains to bus routes, pickup times, bus contact details, or bus route numbers, please provide the URL of the relevant section that matches the answer from the text below.
            source:"". if the context doesn't have any url provide empty source strictly avoid external content or answers
            """,
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


def AzureCosmos_AIQA(human, session_id):
    try:
        start_time = time.time()
        human = human.lower()
        human = human.replace("news", "information")
        human = human.replace("latest", "")
        # Define the pattern to match 'be' as a whole word
        pattern = r'\bbe\b'
        # Replace 'be' with 'biologicale'
        human = re.sub(pattern, 'biologicale', human)
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
            category = response["context"][0].metadata['category']
            if source_links:
                source_link = source_links[0]
            else:
                source_link = ''
                # response = "Sorry, I don't have information. Could you please provide more precise question"
            if category == 'BusPageMenu' or source_links[
                0] == "https://beep.biologicale.com/NavigateUrl?path=LinkPage-SHAMEERPET_INTER_OFFICE_SHUTTLE_SERVICE_TIMMINGS":
                source_input = "context: {} \n\n question : {}".format(response["context"], human)
                source = handle_source(source_input)
                url_pattern = r'https?://[^\s)]+'
                urls = re.findall(url_pattern, source)
                if urls:
                    source_link = urls[0]
            context = [doc.page_content for doc in response["context"]]
            response = response["answer"]
            print(response)
            similarity = text_similarity(str(context), response)
            print(similarity)
            print("\n\n\n")
            if "<table>" not in response:
                if similarity < 0.030:
                    print("score mismatched", similarity)
                    source_link = ''
                    response = "Sorry, I don't have information. Could you please provide more precise question"
            source_link = re.sub(r'.*Files', '', source_link)
            source_link = source_link.replace("D:\\Webapplication\\BEEP\\", "")
            source_link = source_link.replace("C:\\Users\\BRADSOL\\Documents\\Vinoth\\Test\\", "")
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
