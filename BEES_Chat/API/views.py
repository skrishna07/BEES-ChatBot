import datetime
import json
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .SourceCode.BEES_QA import AzureCosmosQA
from .SourceCode.BEES_QA_AI import AzureCosmos_AIQA
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Initialize Cosmos Chat History Container:
client = CosmosClient(os.getenv('WebChat_EndPoint'), os.getenv('WebChat_Key'))
database = client.get_database_client(os.getenv('WebChat_DB'))
History_container = database.get_container_client(os.getenv('WebChat_History_Container'))
History_container = database.create_container_if_not_exists(
    id=os.getenv('WebChat_History_Container'),
    partition_key=PartitionKey(path="/id"),
    offer_throughput=400
)


def get_ip_details(ip, container):
    try:
        query_str = f"SELECT * FROM c WHERE c.ip_address = '{ip}'"
        items = list(container.query_items(query=query_str, enable_cross_partition_query=True))
    except:
        items = []
    if items:
        response = items
        return response
    else:
        return {}


def get_session_details(session_id, container):
    try:
        query_str = f"SELECT c.responses  FROM c WHERE c.session_id = '{session_id}'"
        items = list(container.query_items(query=query_str, enable_cross_partition_query=True))
    except:
        items = []
    if items:
        response = items
        return response
    else:
        return {}


def update_chat_history(container, response, session_id, ip_address):
    # Try to find an existing record with the same session_id
    try:
        query_str = f"SELECT * FROM c WHERE c.session_id = '{session_id}'"
        items = list(container.query_items(query=query_str, enable_cross_partition_query=True))
    except:
        items = []

    if items:
        # Update the existing record
        existing_record = items[0]
        if 'responses' not in existing_record:
            existing_record['responses'] = []
        existing_record['responses'].append(response)
        print("exist response")
        # Replace the item in the container
        container.replace_item(item=existing_record['id'], body=existing_record)
    else:
        # Create a new record
        chat_history_item = {
            'id': str(session_id),
            'ip_address': ip_address,
            'session_id': session_id,
            'responses': [response],
            'datetime': str(datetime.datetime.now())
        }
        container.create_item(body=chat_history_item)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def AISearchResponse(request):
    try:
        print("AI Search")
        request_body = json.loads(request.body)
        query = request_body.get('query')
        session_id = request_body.get('session_id')
        ip_address = request_body.get('ip_address')
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        if not query:
            return Response({"error": "Query not provided"}, status=400)

        response_data, token_used, total_cost, source = AzureCosmos_AIQA(query, session_id)
        response_dict_history = {'query': query, 'response': response_data, 'source': source,
                                 'status': 'success',
                                 'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        response_dict = {'response': response_data, 'source': source, 'session_id': session_id,
                         'status': 'success',
                         'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        # Save to Cosmos DB if response is successful
        update_chat_history(History_container, response_dict_history, session_id, ip_address)
        return Response(response_dict)

    except json.JSONDecodeError as e:
        return Response({"error": "Invalid JSON" + "\n e", 'status': 'Fail', 'statuscode': 400}, status=400)
    except exceptions.CosmosHttpResponseError as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)

    except Exception as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def AIResponse(request):
    try:
        request_body = json.loads(request.body)
        query = request_body.get('query')
        session_id = request_body.get('session_id')
        ip_address = request_body.get('ip_address')
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        if not query:
            return Response({"error": "Query not provided"}, status=400)

        response_data, token_used, total_cost, source = AzureCosmosQA(query, session_id)
        response_dict_history = {'query': query, 'response': response_data, 'source': source,
                                 'status': 'success',
                                 'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        response_dict = {'response': response_data, 'source': source, 'session_id': session_id,
                         'status': 'success',
                         'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        # Save to Cosmos DB if response is successful
        update_chat_history(History_container, response_dict_history, session_id, ip_address)
        return Response(response_dict)

    except json.JSONDecodeError as e:
        return Response({"error": "Invalid JSON" + "\n e", 'status': 'Fail', 'statuscode': 400}, status=400)
    except exceptions.CosmosHttpResponseError as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)

    except Exception as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def IPDetails(request):
    try:
        request_body = json.loads(request.body)
        ip = request_body.get('ip_address')
        response = get_ip_details(ip, History_container)
        return Response(response)

    except json.JSONDecodeError as e:
        return Response({"error": "Invalid JSON" + "\n e", 'status': 'Fail', 'statuscode': 400}, status=400)
    except exceptions.CosmosHttpResponseError as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)
    except Exception as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def SessionDetails(request):
    try:
        request_body = json.loads(request.body)
        session_id = request_body.get('session_id')
        response = get_session_details(session_id, History_container)
        return Response(response)

    except json.JSONDecodeError as e:
        return Response({"error": "Invalid JSON" + "\n e", 'status': 'Fail', 'statuscode': 400}, status=400)
    except exceptions.CosmosHttpResponseError as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)
    except Exception as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)
